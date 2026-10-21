"""The editor's Update posts the saved node to /api/v1/custom_component and replaces it with the response."""

import inspect
import json
import uuid
from pathlib import Path

import pytest
from fastapi import status
from httpx import AsyncClient
from lfx.components.input_output.text import TextInputComponent
from lfx.components.models_and_agents.prompt import PromptComponent
from lfx.custom.validate import extract_class_name
from lfx.graph import Graph

STARTER_FLOWS_V1_11_6 = Path(__file__).parents[5] / "lfx" / "tests" / "fixtures" / "starter_flows" / "v1.11.6"
STARTER_PROJECTS = Path(__file__).parents[4] / "base" / "langflow" / "initial_setup" / "starter_projects"

SEO_V1_11_6 = STARTER_FLOWS_V1_11_6 / "seo_keyword_generator.json"
TWITTER_V1_11_6 = STARTER_FLOWS_V1_11_6 / "twitter_thread_generator.json"
TWITTER_STARTER = STARTER_PROJECTS / "Twitter Thread Generator.json"
SEO_STARTER = STARTER_PROJECTS / "SEO Keyword Generator.json"

# Flows saved before nodes named their component class carry only their code.
SEO_FLOWS = [
    pytest.param(SEO_V1_11_6, id="seo-keyword-generator-v1.11.6"),
    pytest.param(SEO_STARTER, id="seo-keyword-generator-starter"),
]

# Saved code that parsing would cache, blow up in memory, or fail on with MemoryError.
SAVED_CODE_HEADER = "from lfx.custom import Component\nclass PromptComponent(Component):\n    pass\n"
SAVED_CODE_PROBES = [
    pytest.param(SAVED_CODE_HEADER + "#" * 2_000_000, id="2-mb-comment"),
    pytest.param(SAVED_CODE_HEADER + "a;" * 100_000, id="many-statements"),
    pytest.param(SAVED_CODE_HEADER + "x = " + "-" * 10_000 + "1\n", id="deeply-nested"),
]

PROMPT_INPUT_CODE = """
from lfx.custom.custom_component.component import Component
from lfx.io import Output, PromptInput
from lfx.schema.message import Message


class GreetingPrompt(Component):
    display_name = "Greeting Prompt"
    inputs = [PromptInput(name="template", display_name="Template")]
    outputs = [Output(display_name="Prompt", name="prompt", method="build_prompt")]

    async def build_prompt(self) -> Message:
        return await Message.from_template_and_variables(**self._attributes)
"""


def _saved_prompt(flow_path: Path) -> dict:
    flow = json.loads(flow_path.read_text(encoding="utf-8"))
    return next(node for node in flow["data"]["nodes"] if node["data"]["type"] == "Prompt")["data"]


async def _editor_update(client: AsyncClient, headers: dict, saved_node: dict, component=PromptComponent) -> dict:
    code = Path(inspect.getsourcefile(component)).read_text(encoding="utf-8")
    response = await client.post(
        "api/v1/custom_component", json={"code": code, "frontend_node": saved_node}, headers=headers
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    return response.json()["data"]


@pytest.mark.parametrize(
    "flow_path",
    [
        pytest.param(SEO_V1_11_6, id="seo-keyword-generator-v1.11.6"),
        pytest.param(TWITTER_V1_11_6, id="twitter-thread-generator-v1.11.6"),
        pytest.param(TWITTER_STARTER, id="twitter-thread-generator-starter"),
    ],
)
async def test_editor_update_keeps_prompt_variables_and_values(
    client: AsyncClient, logged_in_headers: dict, flow_path: Path
):
    saved = _saved_prompt(flow_path)["node"]
    variables = saved["custom_fields"]["template"]
    assert len(variables) == 6
    assert all(saved["template"][name]["value"] for name in variables)

    updated = await _editor_update(client, logged_in_headers, saved)

    assert sorted(updated["custom_fields"]["template"]) == sorted(variables)
    assert {name: updated["template"].get(name, {}).get("value") for name in variables} == {
        name: saved["template"][name]["value"] for name in variables
    }


@pytest.mark.parametrize(
    "flow_path",
    [
        pytest.param(TWITTER_V1_11_6, id="twitter-thread-generator-v1.11.6"),
        pytest.param(TWITTER_STARTER, id="twitter-thread-generator-starter"),
    ],
)
async def test_updated_twitter_thread_prompt_still_runs(client: AsyncClient, logged_in_headers: dict, flow_path: Path):
    """Its template uses {PROFILE_TYPE} as well as {{PROFILE_TYPE}}, so the current parser finds no variables."""
    saved = _saved_prompt(flow_path)
    updated = await _editor_update(client, logged_in_headers, saved["node"])
    node_id = saved["id"]
    graph = Graph.from_payload(
        {"nodes": [{"id": node_id, "type": "genericNode", "data": {**saved, "node": updated}}], "edges": []},
        flow_id="twitter-thread-generator",
    )

    async for _ in graph.async_start():
        pass

    prompt = graph.get_vertex(node_id).built_object["prompt"]
    assert saved["node"]["template"]["PROFILE_TYPE"]["value"] in prompt.text


async def test_editor_update_keeps_variables_of_a_template_the_validator_rejects(
    client: AsyncClient, logged_in_headers: dict
):
    """The Prompt's hook lists no variables when it rejects the template, but they are still the user's."""
    saved = _saved_prompt(SEO_STARTER)["node"]
    saved["template"]["template"]["value"] += " {bad var}"
    variables = saved["custom_fields"]["template"]

    updated = await _editor_update(client, logged_in_headers, saved)

    assert updated["custom_fields"]["template"] == variables
    assert {name: updated["template"][name]["value"] for name in variables} == {
        name: saved["template"][name]["value"] for name in variables
    }


async def test_code_editor_save_keeps_the_variables_of_a_users_prompt_input(
    client: AsyncClient, logged_in_headers: dict
):
    """Only the core Prompt's hook rebuilds variables, so a user component with a PromptInput relies on the save."""
    response = await client.post("api/v1/custom_component", json={"code": PROMPT_INPUT_CODE}, headers=logged_in_headers)
    assert response.status_code == status.HTTP_200_OK, response.text
    node = response.json()["data"]
    response = await client.post(
        "api/v1/validate/prompt",
        json={"name": "template", "template": "Hi {name} from {city}", "frontend_node": node},
        headers=logged_in_headers,
    )
    assert response.status_code == status.HTTP_200_OK, response.text
    saved = response.json()["frontend_node"]
    assert saved["metadata"]["component_class_name"] == "GreetingPrompt"
    saved["template"]["name"]["value"] = "Ana"
    saved["template"]["city"]["value"] = "Recife"

    response = await client.post(
        "api/v1/custom_component",
        json={"code": PROMPT_INPUT_CODE + "\n# edited\n", "frontend_node": saved},
        headers=logged_in_headers,
    )

    assert response.status_code == status.HTTP_200_OK, response.text
    updated = response.json()["data"]
    assert updated["custom_fields"]["template"] == ["name", "city"]
    assert updated["template"]["name"]["value"] == "Ana"
    assert updated["template"]["city"]["value"] == "Recife"


@pytest.mark.parametrize("flow_path", SEO_FLOWS)
async def test_code_swapped_for_another_component_drops_the_prompt_variables(
    client: AsyncClient, logged_in_headers: dict, flow_path: Path
):
    """The code editor posts to the same route, so the new component must not inherit the Prompt's variables."""
    saved = _saved_prompt(flow_path)["node"]

    updated = await _editor_update(client, logged_in_headers, saved, component=TextInputComponent)

    assert not set(saved["custom_fields"]["template"]) & set(updated["template"])
    assert not any(updated.get("custom_fields", {}).values())


@pytest.mark.parametrize("saved_code", SAVED_CODE_PROBES)
async def test_editor_update_never_parses_the_saved_code(client: AsyncClient, logged_in_headers: dict, saved_code: str):
    """The saved node comes from the client, so its code is not parsed (nor cached) to tell which component it is."""
    saved = _saved_prompt(SEO_V1_11_6)["node"]
    variables = saved["custom_fields"]["template"]
    await _editor_update(client, logged_in_headers, saved)
    saved["template"]["code"]["value"] = saved_code
    misses = extract_class_name.cache_info().misses

    updated = await _editor_update(client, logged_in_headers, saved)

    assert extract_class_name.cache_info().misses == misses
    assert {name: updated["template"][name]["value"] for name in variables} == {
        name: saved["template"][name]["value"] for name in variables
    }


async def test_re_added_custom_field_clears_another_users_file(client: AsyncClient, logged_in_headers: dict):
    """A custom field the update lacks goes through the same file_path check as any other field."""
    saved = _saved_prompt(SEO_V1_11_6)["node"]
    saved["custom_fields"]["template"].append("doc")
    saved["template"]["doc"] = {"type": "file", "value": "secret.pdf", "file_path": f"{uuid.uuid4()}/secret.pdf"}

    updated = await _editor_update(client, logged_in_headers, saved)

    assert "doc" in updated["custom_fields"]["template"]
    assert updated["template"]["doc"]["value"] == ""
    assert updated["template"]["doc"]["file_path"] == ""
