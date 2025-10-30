# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# imports for structured outputs tests
from typing import Optional

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
import requests

from ..utils import RemoteOpenAIServer

# any model with a chat template should work here
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
CLOSE_BADREQUEST_CASES = [
    (
        "nonexistent_session_id",
        {"session_id": "nonexistent-session-id"},
        "session not found"
    ),
    (
        "malformed_close_request",
        {"body": {"extra-field": "extra-field-data"}},
        None
    )
]


@pytest.fixture(scope="module")
def server(zephyr_lora_files):  # noqa: F811
    args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "8192",
        "--enforce-eager",
        # lora config below
        "--enable-lora",
        "--lora-modules",
        f"zephyr-lora={zephyr_lora_files}",
        "--max-lora-rank",
        "64",
        "--max-cpu-loras",
        "2",
        "--max-num-seqs",
        "128",
    ]

    with RemoteOpenAIServer(MODEL_NAME, args) as remote_server:
        yield remote_server


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
async def test_create_session_badrequest(server: RemoteOpenAIServer):
    bad_response = requests.post(
        server.url_for("invocations"),
        json={"requestType": "NEW_SESSION", "extra-field": "extra-field-data"}
    )

    assert bad_response.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("test_name,request_change,expected_error", CLOSE_BADREQUEST_CASES)
async def test_close_session_badrequest(
    server: RemoteOpenAIServer,
    test_name: str,
    request_change: dict,
    expected_error: Optional[str],
):
    # first attempt to create a session
    url = server.url_for("invocations")
    create_response = requests.post(
        url,
        json={"requestType": "NEW_SESSION"}
    )
    create_response.raise_for_status()
    valid_session_id = create_response.headers.get("X-Amzn-SageMaker-New-Session-Id")
    assert valid_session_id is not None

    close_request_json = {"requestType": "CLOSE"}
    if request_change.get("body"):
        close_request_json.update(request_change.get("body"))
    bad_session_id = request_change.get("session_id")
    bad_close_response = requests.post(
        url,
        headers={"X-Amzn-SageMaker-Session-Id": bad_session_id or valid_session_id},
        json=close_request_json
    )

    # clean up created session, should succeed
    clean_up_response = requests.post(
        url,
        headers={"X-Amzn-SageMaker-Session-Id": valid_session_id},
        json={"requestType": "CLOSE"}
    )
    clean_up_response.raise_for_status()

    assert bad_close_response.status_code == 400
    if expected_error:
        assert expected_error in bad_close_response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_close_session_invalidrequest(server: RemoteOpenAIServer, client: openai.AsyncOpenAI):
    # first attempt to create a session
    url = server.url_for("invocations")
    create_response = requests.post(
        url,
        json={"requestType": "NEW_SESSION"}
    )
    create_response.raise_for_status()
    valid_session_id = create_response.headers.get("X-Amzn-SageMaker-New-Session-Id")
    assert valid_session_id is not None

    close_request_json = {"requestType": "CLOSE"}
    invalid_close_response = requests.post(
        url,
        # no headers to specify session_id
        json=close_request_json
    )

    # clean up created session, should succeed
    clean_up_response = requests.post(
        url,
        headers={"X-Amzn-SageMaker-Session-Id": valid_session_id},
        json={"requestType": "CLOSE"}
    )
    clean_up_response.raise_for_status()

    assert invalid_close_response.status_code == 424
    assert "invalid session_id" in invalid_close_response.json()["error"]["message"]


@pytest.mark.asyncio
async def test_session(server: RemoteOpenAIServer, client: openai.AsyncOpenAI):
    # first attempt to create a session
    url = server.url_for("invocations")
    create_response = requests.post(
        url,
        json={"requestType": "NEW_SESSION"}
    )
    create_response.raise_for_status()
    valid_session_id = create_response.headers.get("X-Amzn-SageMaker-New-Session-Id")
    assert valid_session_id is not None

    # test invocation with session id
    messages = [
        {"role": "system", "content": "you are a helpful assistant"},
        {"role": "user", "content": "what is 1+1?"},
    ]

    request_args = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_completion_tokens": 5,
        "temperature": 0.0,
        "logprobs": False,
    }

    chat_completion = await client.chat.completions.create(**request_args)

    invocation_response = requests.post(
        server.url_for("invocations"),
        headers={"X-Amzn-SageMaker-Session-Id": valid_session_id},
        json=request_args
    )
    invocation_response.raise_for_status()

    chat_output = chat_completion.model_dump()
    invocation_output = invocation_response.json()

    # close created session, should succeed
    close_response = requests.post(
        url,
        headers={"X-Amzn-SageMaker-Session-Id": valid_session_id},
        json={"requestType": "CLOSE"}
    )
    close_response.raise_for_status()

    assert chat_output.keys() == invocation_output.keys()
    assert chat_output["choices"] == invocation_output["choices"]
    assert close_response.headers.get("X-Amzn-SageMaker-Closed-Session-Id") == valid_session_id
