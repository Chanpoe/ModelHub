import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
import tiktoken

from modelhub.context import OpenAIContext

load_dotenv()


class Dialog(ABC):
    def __init__(self, model_name: str, system_prompt: str = ""):
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

    def get_token_count(self) -> int:
        """
        计算当前上下文中的token使用量
        :return: token总数
        """
        # 初始化tiktoken编码器
        encoding = tiktoken.encoding_for_model(self.model_name)
        total_tokens = 0

        # 遍历所有消息
        for message in self.context.messages:
            # 处理文本消息
            if message.get("role"):
                # 计算角色名的token
                total_tokens += len(encoding.encode(message["role"]))
                # 计算内容的token
                if isinstance(message.get("content"), str):
                    total_tokens += len(encoding.encode(message["content"]))

            # 处理图片消息
            if message.get("type") == "image_url_list":
                # 图片消息按照OpenAI的计算方式
                # 高清图片消费65元/百万token，标准图片消费8.5元/百万token
                # 这里按照标准图片计算，每张图片算作85个token
                total_tokens += 85

        return total_tokens

    def send(self, message: str = None, base64_image_list=None, image_url_list=None):
        """
        发送消息到模型并获取回复
        :param message: 用户发送的消息
        :param base64_image_list:
        :param image_url_list:
        :return: 模型的回复
        """
        # 添加图片消息到上下文
        if base64_image_list or image_url_list:
            if base64_image_list:
                self.context.add_image_message(base64_image_list=base64_image_list)
            if image_url_list:
                self.context.add_image_message(image_url_list=image_url_list)

        # 如果有新消息才添加
        if message:
            # 添加用户消息到上下文
            self.context.add_user_message(message)

        chat_completion = self.client.chat.completions.create(
            messages=self.context.messages,
            model=self.model_name,
            top_p=self.top_p,
            temperature=self.temperature,
            # response_format="json",
            # stream=True
        )
        try:
            res = chat_completion.choices[0].message.content
        except:
            res = ''
        # for trunk in chat_completion:
        #     res += trunk.choices[0].delta.content
        # print()

        # 添加模型回复到上下文
        self.context.add_assistant_message(res)
        return res


class OpenRouterDialog(Dialog):
    def __init__(self, model_name: str, system_prompt: str = ""):
        super().__init__(model_name, system_prompt)
        self._context = OpenAIContext(system_prompt=system_prompt)
        # 初始化OpenAI同步客户端
        self._client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )

    @property
    def context(self):
        return self._context

    @property
    def client(self):
        return self._client


class OpenAIDialog(Dialog):
    def __init__(self, model_name: str, system_prompt: str = ""):
        super().__init__(model_name, system_prompt)
        self._context = OpenAIContext(system_prompt=system_prompt)
        # 初始化OpenAI异步客户端
        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

    @property
    def context(self):
        return self._context

    @property
    def client(self):
        return self._client


class GPTsAPIDialog(Dialog):
    def __init__(self, model_name: str, system_prompt: str = ""):
        super().__init__(model_name, system_prompt)
        self._context = OpenAIContext(system_prompt=system_prompt)
        # 初始化OpenAI异步客户端
        self._client = OpenAI(
            api_key=os.getenv("GPTS_API_KEY"),
            base_url="https://api.gptsapi.net/v1"
        )
        self.temperature = 1  # GPTs API使用openai o3模型时候只能设置 temperature = 1

    @property
    def context(self):
        return self._context

    @property
    def client(self):
        return self._client


class VolcDialog(Dialog):
    def __init__(self, model_name: str, system_prompt: str = ""):
        super().__init__(model_name, system_prompt)
        self._context = OpenAIContext(system_prompt=system_prompt)
        # 初始化OpenAI异步客户端
        self._client = OpenAI(
            api_key=os.getenv("VOLC_API_KEY"),
            base_url="https://ark.cn-beijing.volces.com/api/v3/"
        )
        self.temperature = 1  # GPTs API使用openai o3模型时候只能设置 temperature = 1

    @property
    def context(self):
        return self._context

    @property
    def client(self):
        return self._client


class DMXDialog(Dialog):
    def __init__(self, model_name: str, system_prompt: str = "", area='cn'):
        """
        DMX 对话类，兼容 OpenAI SDK
        :param model_name:
        :param system_prompt:
        :param area: 地区，默认cn
        """
        super().__init__(model_name, system_prompt)
        self._context = OpenAIContext(system_prompt=system_prompt)
        # 初始化OpenAI同步客户端
        self._client = OpenAI(
            api_key=os.getenv("DMX_API_KEY"),
            base_url="https://www.dmxapi.cn/v1" if area == 'cn' else "https://www.dmxapi.com/v1"
        )

    @property
    def context(self):
        return self._context

    @property
    def client(self):
        return self._client


if __name__ == '__main__':
    # 测试代码
    # dialog = OpenRouterDialog(model_name="openai/o4-mini-high", system_prompt="解答用户问题")
    # dialog = OpenAIDialog(model_name="o4-mini-2025-04-16", system_prompt="解答用户问题")
    # dialog = GPTsAPIDialog(model_name="o4-mini", system_prompt="解答用户问题")
    dialog = VolcDialog(model_name='doubao-pro-256k-241115', system_prompt='解答用户问题')
    # 添加图片消息
    import base64

    # with open('test.jpeg', 'rb') as f:
    #     image_bytes = f.read()
    #     pic_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # dialog.context.add_image_message(text_content='但丁真不是中国人但丁真是中国人，这句话怎么理解', base64_image=pic_base64)
    dialog.context.add_user_message(content='但丁真不是中国人但丁真是中国人，这句话怎么理解')
    dialog.send()
    print(dialog.context.messages)
