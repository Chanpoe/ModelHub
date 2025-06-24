from typing import List, Dict


# 上下文基类
class Context:
    def __init__(self, system_prompt: str):
        # 历史消息
        self.messages: List[Dict] = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]

    def add_user_message(self, content: str):
        """
        添加一条用户发送的消息
        :param content: 消息文本
        :return:
        """
        raise NotImplementedError("添加用户消息的方法未实现")

    def add_assistant_message(self, content: str):
        """
        添加一条AI回复的消息
        :param content: 消息文本
        :return:
        """
        raise NotImplementedError("添加AI消息的方法未实现")

    def add_image_message(self, text_content: str, base64_image=None, image_url=None):
        """
        添加一条图片消息
        :param text_content: 文本内容
        :param base64_image: 图片的base64编码
        :param image_url: 图片的URL地址
        :return:
        """
        pass

    # 清除对话历史记录
    def clear_history(self):
        self.messages = [
            {
                "role": "system",
                "content": self.messages[0]["content"],
            }
        ]


# OpenAI API兼容 上下文类
class OpenAIContext(Context):
    def __init__(self, system_prompt: str = ""):
        super().__init__(system_prompt)
        # 设置默认历史消息
        self.clear_history()

    def add_user_message(self, content: str):
        self.messages.append({
            "role": "user",
            "content": content,
        })

    def add_assistant_message(self, content: str):
        self.messages.append({
            "role": "assistant",
            "content": content,
        })

    def add_image_message(self, text_content: str, base64_image_list: list = None, image_url_list: list = None):
        if base64_image_list is None and image_url_list is None:
            raise ValueError("必须提供 base64_image_list 或 image_url_list")
        if base64_image_list is not None:
            content_ = [{"type": "text", "text": str(text_content)}]
            for base64_image in base64_image_list:
                content_.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}",
                        "detail": "low"
                    }
                })
            self.messages.append({
                "role": "user",
                "content": content_,
            })

        elif image_url_list is not None:
            content_ = [{"type": "text", "text": str(text_content)}]
            for image_url in image_url_list:
                content_.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    }
                })
            self.messages.append({
                "role": "user",
                "content": content_,
            })
