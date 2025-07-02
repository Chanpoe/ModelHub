import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
import tiktoken

from modelhub.context import OpenAIContext

load_dotenv()


class Dialog(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.api_key = None
        self.top_p = 1
        self.temperature = 0.3

    @property
    @abstractmethod
    def context(self) -> OpenAIContext:
        """
        对话使用的上下文对象
        :return: 对话上下文实例
        """
        pass

    @property
    @abstractmethod
    def client(self) -> OpenAI:
        """
        对话使用的客户端
        :return:
        """
        pass

    @property
    @abstractmethod
    def async_client(self) -> AsyncOpenAI:
        """
        异步对话使用的客户端
        :return: 异步OpenAI客户端实例
        """
        pass

    def get_token_count(self) -> int:
        """
        计算当前上下文中的token使用量
        :return: token总数
        """
        encoding = tiktoken.encoding_for_model(self.model_name)
        total_tokens = 0
        for message in self.context.messages:
            if message.get("role"):
                total_tokens += len(encoding.encode(message["role"]))
                if isinstance(message.get("content"), str):
                    total_tokens += len(encoding.encode(message["content"]))
            # 更健壮地处理图片消息
            if isinstance(message.get("content"), list):
                for item in message["content"]:
                    if item.get("type") == "image_url":
                        total_tokens += 85
        return total_tokens

    def send(self, message: str = "", base64_image_list: list = None, image_url_list: list = None) -> str:
        """
        发送消息到模型并获取回复
        :param message: 用户发送的消息
        :param base64_image_list:
        :param image_url_list:
        :return: 模型的回复
        """
        # 如果新消息有图片
        if base64_image_list or image_url_list:
            self.context.add_image_message(
                text_content=message,
                base64_image_list=base64_image_list,
                image_url_list=image_url_list
            )
        # 如果有新消息且没有图片，添加用户消息到上下文
        if message and not base64_image_list and not image_url_list:
            self.context.add_user_message(content=message)
        messages = []
        for m in self.context.messages:
            if isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})
            elif isinstance(m.get("content"), list):
                messages.append({"role": m["role"], "content": m["content"]})
        # 检查API KEY
        if not self.client.api_key:
            raise RuntimeError(f"API KEY未设置，请检查环境变量。当前模型: {self.model_name}")
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                top_p=self.top_p,
                temperature=self.temperature,
            )
            res = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] 获取模型回复失败: {e}")
            res = ''
        self.context.add_assistant_message(res)
        return res

    async def async_send(self, message: str = None, base64_image_list: list = None, image_url_list: list = None) -> str:
        """
        异步发送消息到模型并获取回复
        :param message: 用户发送的消息
        :param base64_image_list:
        :param image_url_list:
        :return: 模型的回复
        """
        if base64_image_list or image_url_list:
            self.context.add_image_message(
                text_content=message,
                base64_image_list=base64_image_list,
                image_url_list=image_url_list
            )
        if message and not base64_image_list and not image_url_list:
            self.context.add_user_message(content=message)
        messages = []
        for m in self.context.messages:
            if isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})
            elif isinstance(m.get("content"), list):
                messages.append({"role": m["role"], "content": m["content"]})
        if not self.async_client.api_key:
            raise RuntimeError(f"API KEY未设置，请检查环境变量。当前模型: {self.model_name}")
        try:
            chat_completion = await self.async_client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                top_p=self.top_p,
                temperature=self.temperature,
            )
            res = chat_completion.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] 获取模型回复失败: {e}")
            res = ''
        self.context.add_assistant_message(res)
        return res


class GenericDialog(Dialog):
    def __init__(self, model_name: str, system_prompt: str = "", api_key_env: str = None, base_url: str = None):
        super().__init__(model_name)
        self._context = OpenAIContext(system_prompt=system_prompt)
        api_key = os.getenv(api_key_env) if api_key_env else None
        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key)
            self._async_client = AsyncOpenAI(api_key=api_key)

    @property
    def context(self):
        return self._context

    @property
    def client(self):
        return self._client

    @property
    def async_client(self):
        return self._async_client


class OpenRouterDialog(GenericDialog):
    def __init__(self, model_name: str, system_prompt: str = ""):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1"
        )


class OpenAIDialog(GenericDialog):
    def __init__(self, model_name: str, system_prompt: str = ""):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            api_key_env="OPENAI_API_KEY"
        )


class VolcDialog(GenericDialog):
    def __init__(self, model_name: str, system_prompt: str = ""):
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            api_key_env="VOLC_API_KEY",
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )


class DMXDialog(GenericDialog):
    def __init__(self, model_name: str, system_prompt: str = "", area: str = 'cn'):
        base_url = "https://www.dmxapi.cn/v1" if area == 'cn' else "https://www.dmxapi.com/v1"
        super().__init__(
            model_name=model_name,
            system_prompt=system_prompt,
            api_key_env="DMX_API_KEY",
            base_url=base_url
        )


if __name__ == '__main__':
    # 测试代码
    dialog = VolcDialog(model_name='doubao-pro-256k-241115', system_prompt='解答用户问题')
    dialog.context.add_user_message(content='但丁真不是中国人但丁真是中国人，这句话怎么理解')
    dialog.send()
    print(dialog.context.messages)
