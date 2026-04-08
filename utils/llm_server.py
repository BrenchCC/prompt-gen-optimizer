import logging
import os
import sys

from typing import Any, Dict, List, Optional

from openai import OpenAI
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

load_dotenv()

sys.path.append(os.getcwd())

logger = logging.getLogger("LLM-Client")


class LLM_Client:
    def __init__(
        self,
        mode: str = "ark",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: int = 300,
        print_stream: bool = True,
    ) -> None:
        mode_norm = (mode or "").strip().lower()
        if mode_norm in {"volcengine", "ark"}:
            self.mode = "ark"
        elif mode_norm in {"openai", "oai"}:
            self.mode = "openai"
        else:
            raise ValueError(f"Unsupported LLM mode: {mode!r}")

        if self.mode == "ark":
            resolved_api_key = api_key or os.environ.get("LLM_API_KEY")
            resolved_base_url = base_url or os.environ.get("LLM_API_BASE_URL")
        else:
            resolved_api_key = (
                api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
            )
            resolved_base_url = (
                base_url
                or os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE_URL")
                or os.environ.get("LLM_API_BASE_URL")
            )

        if not resolved_api_key:
            raise RuntimeError("Missing API key in environment or parameters.")

        self.api_key = resolved_api_key
        self.base_url = resolved_base_url
        self.default_model = default_model
        self.timeout = timeout
        self.print_stream = print_stream

        if self.mode == "ark":
            self._client = Ark(base_url=self.base_url, api_key=self.api_key)
        else:
            if self.base_url:
                self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            else:
                self._client = OpenAI(api_key=self.api_key)

    def chat(
        self,
        input_query: str,
        end_point: Optional[str] = None,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        reasoning_option: Any = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ):
        model = end_point or self.default_model
        if not model:
            raise ValueError("Missing model/end_point.")

        if messages is None:
            assembled_messages = [{"role": "user", "content": input_query}]
            if system_prompt:
                assembled_messages = (
                    [{"role": "system", "content": system_prompt}] + assembled_messages
                )
        else:
            assembled_messages = messages

        resolved_extra_body = extra_body
        if resolved_extra_body is None:
            if self.mode == "ark":
                if isinstance(reasoning_option, dict):
                    resolved_extra_body = reasoning_option
                elif isinstance(reasoning_option, str):
                    resolved_extra_body = {"thinking": {"type": reasoning_option}}
                elif isinstance(reasoning_option, bool):
                    resolved_extra_body = {
                        "thinking": {"type": "enabled" if reasoning_option else "disabled"}
                    }
            elif self.mode == "openai":
                if isinstance(reasoning_option, dict):
                    resolved_extra_body = reasoning_option
                elif isinstance(reasoning_option, str):
                    if reasoning_option in {"enabled", "on", "true"}:
                        effort = "low"
                    elif reasoning_option in {"disabled", "off", "false"}:
                        effort = "none"
                    else:
                        effort = reasoning_option
                    resolved_extra_body = {"reasoning": {"effort": effort}}
                elif isinstance(reasoning_option, bool):
                    if reasoning_option:
                        resolved_extra_body = {"reasoning": {"effort": "low"}}
                    else:
                        model_norm = str(model).strip().lower()
                        if model_norm.startswith(("gpt-5", "o")):
                            resolved_extra_body = {"reasoning": {"effort": "none"}}

        create_kwargs = {
            "model": model,
            "messages": assembled_messages,
            "timeout": timeout if timeout is not None else self.timeout,
            "stream": stream,
        }
        if temperature is not None:
            create_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            create_kwargs["top_p"] = float(top_p)
        if resolved_extra_body is not None:
            create_kwargs["extra_body"] = resolved_extra_body

        try:
            if (
                self.mode == "openai"
                and isinstance(resolved_extra_body, dict)
                and "reasoning" in resolved_extra_body
            ):
                try:
                    reasoning = resolved_extra_body.get("reasoning")
                    extra_passthrough = dict(resolved_extra_body)
                    extra_passthrough.pop("reasoning", None)

                    response_kwargs = {
                        "model": model,
                        "input": assembled_messages,
                        "reasoning": reasoning,
                        "timeout": timeout if timeout is not None else self.timeout,
                        "stream": stream,
                    }
                    if temperature is not None:
                        response_kwargs["temperature"] = float(temperature)
                    if top_p is not None:
                        response_kwargs["top_p"] = float(top_p)
                    if extra_passthrough:
                        response_kwargs["extra_body"] = extra_passthrough

                    response = self._client.responses.create(**response_kwargs)
                    if stream:
                        result = ""
                        for event in response:
                            delta = getattr(event, "delta", None)
                            if isinstance(delta, str) and delta:
                                logger.info(delta)
                                result += delta
                                if self.print_stream:
                                    print(delta, end="")
                            else:
                                text = getattr(event, "text", None)
                                if isinstance(text, str) and text:
                                    logger.info(text)
                                    result += text
                                    if self.print_stream:
                                        print(text, end="")
                    else:
                        result = getattr(response, "output_text", None)
                        if not result and hasattr(response, "output"):
                            output = getattr(response, "output", None)
                            if isinstance(output, list):
                                result = "".join(
                                    str(item.get("text", ""))
                                    for item in output
                                    if isinstance(item, dict) and item.get("type") == "output_text"
                                )

                    reasoning_content = ""
                    prompt_tok, completion_tok = "", ""
                    usage = getattr(response, "usage", None)
                    if usage is not None:
                        prompt_tok = getattr(usage, "input_tokens", "")
                        completion_tok = getattr(usage, "output_tokens", "")

                    if not isinstance(result, str):
                        result = "dummy_result"

                    return reasoning_content, result, prompt_tok, completion_tok
                except Exception as e:
                    logger.error(e)
                    fallback_kwargs = dict(create_kwargs)
                    fb_extra = fallback_kwargs.get("extra_body")
                    if isinstance(fb_extra, dict):
                        fb_extra = dict(fb_extra)
                        fb_extra.pop("reasoning", None)
                        if fb_extra:
                            fallback_kwargs["extra_body"] = fb_extra
                        else:
                            fallback_kwargs.pop("extra_body", None)
                    completion = self._client.chat.completions.create(**fallback_kwargs)
            else:
                completion = self._client.chat.completions.create(**create_kwargs)
            if stream:
                result = ""
                for tok in completion:
                    choices = getattr(tok, "choices", None)
                    if not choices:
                        continue
                    delta = getattr(choices[0], "delta", None)
                    content_piece = None
                    if delta is not None:
                        content_piece = getattr(delta, "content", None)
                    if not content_piece:
                        continue
                    logger.info(content_piece)
                    result += content_piece
                    if self.print_stream:
                        print(content_piece, end="")
            else:
                result = completion.choices[0].message.content

            try:
                reasoning_content = completion.choices[0].message.reasoning_content
                prompt_tok = completion.usage.prompt_tokens
                completion_tok = completion.usage.completion_tokens
                # logger.info(f"reasoning_content: {reasoning_content}")
            except Exception as e:
                logger.error(f"Error extracting reasoning_content: {e}")
                reasoning_content = ""
                prompt_tok, completion_tok = "", ""
                logger.info(f"Prompt Tokens: {prompt_tok}, Completion Tokens: {completion_tok}")

            if not stream:
                # logger.info(f"result: {result}")
                # logger.info("No Streaming") 
                pass

            if not isinstance(result, str):
                result = "dummy_result"
                reasoning_content = ""
                prompt_tok, completion_tok = "", ""

            return reasoning_content, result, prompt_tok, completion_tok
        except Exception as e:
            logger.error(e)
            return None, "dummy_result", "", ""

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()],
    )

    client =  LLM_Client(
        mode = "ark",
        api_key = os.environ.get("WORKER_API_KEY"),
        base_url = os.environ.get("WORKER_BASE_URL"),
        default_model = os.environ.get("WORKER_MODEL_NAME"),
    )
    reasoning_content, result, prompt_tok, completion_tok = client.chat(
        input_query = "你好",
        end_point = os.environ.get("WORKER_MODEL_NAME"),
        reasoning_option = False,
    )
    logger.info(result)



    client = LLM_Client(
        mode = "openai",
        api_key = os.environ.get("WORKER_API_KEY"),
        base_url = os.environ.get("WORKER_BASE_URL"),
        default_model = os.environ.get("WORKER_MODEL_NAME"),
    )
    reasoning_content, result, prompt_tok, completion_tok = client.chat(
        input_query = "你好",
        end_point = os.environ.get("WORKER_MODEL_NAME"),
        reasoning_option = False,   
    )
    logger.info(reasoning_content)
    logger.info(result)
