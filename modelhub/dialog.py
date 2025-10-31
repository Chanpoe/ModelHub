import os
import re, json
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

    @staticmethod
    def format_response_output(content: str) -> any:
        """
        提取和解析模型回复中的JSON结构，若解析失败则返回原始内容。
        :param content: 模型回复内容
        :return: 结构化字典或原始内容
        """
        def extract_json(text):
            # 优先提取代码块里的内容
            match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
            if match:
                return match.group(1)
            # 匹配数组或对象
            match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
            if match:
                return match.group(1)
            return text
        json_str = extract_json(content)
        try:
            res = json.loads(json_str)
        except Exception as e:
            print(f"[WARN] JSON解析失败，返回原始内容: {e}")
            res = content
        return res

    def _get_chat_completion(self, messages):
        """
        同步请求模型回复
        """
        return self.client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            top_p=self.top_p,
            temperature=self.temperature,
        )

    async def _get_chat_completion_async(self, messages):
        """
        异步请求模型回复
        """
        return await self.async_client.chat.completions.create(
            messages=messages,
            model=self.model_name,
            top_p=self.top_p,
            temperature=self.temperature,
        )

    def send(self, message: str = "", base64_image_list: list = None, image_url_list: list = None, format_output: bool = False) -> any:
        """
        发送消息到模型，并返回模型回复。
        :param message: 用户消息
        :param base64_image_list: base64编码图片列表
        :param image_url_list: 图片url地址列表
        :param format_output: 是否格式化输出，如果为True，则返回JSON格式，默认返回原始内容
        :return: 模型回复或者JSON格式内容
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
        # 构建消息列表
        messages = []
        if format_output:
            self.context.add_user_message('输出为JSON格式')
        for m in self.context.messages:
            if isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})
            elif isinstance(m.get("content"), list):
                messages.append({"role": m["role"], "content": m["content"]})
        # 检查API KEY
        if not self.client.api_key:
            raise RuntimeError(f"API KEY未设置，请检查环境变量。当前Dialog类: {self.__class__.__name__}, 当前模型: {self.model_name}")
        try:
            chat_completion = self._get_chat_completion(messages)
            content = chat_completion.choices[0].message.content
            if format_output:
                res = self.format_response_output(content)
            else:
                res = content
            self.context.add_assistant_message(str(res))
            return res
        except Exception as e:
            print(f"[ERROR] 获取模型回复失败: {e}")
            res = ''
            self.context.add_assistant_message(res)
            return res

    async def async_send(self, message: str = None, base64_image_list: list = None, image_url_list: list = None, format_output: bool = False) -> any:
        """
        异步发送消息到模型，并返回模型回复。
        :param message: 用户消息
        :param base64_image_list: base64编码图片列表
        :param image_url_list: 图片url地址列表
        :param format_output: 是否格式化输出，如果为True，则返回JSON格式，默认返回原始内容
        :return: 模型回复或者JSON格式内容
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
        # 构建消息列表
        messages = []
        if format_output:
            self.context.add_user_message('输出为JSON格式')
        for m in self.context.messages:
            if isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})
            elif isinstance(m.get("content"), list):
                messages.append({"role": m["role"], "content": m["content"]})
        if not self.async_client.api_key:
            raise RuntimeError(f"API KEY未设置，请检查环境变量。当前Dialog类: {self.__class__.__name__}, 当前模型: {self.model_name}")
        try:
            chat_completion = await self._get_chat_completion_async(messages)
            content = chat_completion.choices[0].message.content
            if format_output:
                res = self.format_response_output(content)
            else:
                res = content
            self.context.add_assistant_message(str(res))
            return res
        except Exception as e:
            print(f"[ERROR] 获取模型回复失败: {e}")
            res = ''
            self.context.add_assistant_message(res)
            return res


class GenericDialog(Dialog):
    def __init__(self, model_name: str, system_prompt: str = "", api_key_env: str = None, base_url: str = None):
        super().__init__(model_name)
        self._context = OpenAIContext(system_prompt=system_prompt)
        # 初始化时候从环境变量获取API Key，但是实例化之后改变这个变量并不能修改client的api_key属性
        _tmp_ak_v = os.getenv(api_key_env) if api_key_env else None
        if base_url:
            self._client = OpenAI(base_url=base_url, api_key=_tmp_ak_v)
            self._async_client = AsyncOpenAI(base_url=base_url, api_key=_tmp_ak_v)
        else:
            self._client = OpenAI(api_key=_tmp_ak_v)
            self._async_client = AsyncOpenAI(api_key=_tmp_ak_v)

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
    # dialog = DMXDialog(model_name="deepseek-ai/DeepSeek-V3", area='en')
    dialog = DMXDialog(model_name='deepseek-v3.1',system_prompt='你是AI助手', area='en')

    # 发送测试消息
    result = dialog.send(
        message="请告诉我iPhone 15的名称和价格，不同的型号用列表列出，直接用中文回答。",
        format_output=True
    )
    print(result)
