"""LLM客户端封装"""
import os
from typing import Optional
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """LLM客户端抽象基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成文本"""
        pass


class OpenAIClient(LLMClient):
    """OpenAI / OpenAI 兼容接口客户端"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        base_url: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        if not self.api_key:
            raise ValueError("需要设置OPENAI_API_KEY环境变量或传入api_key")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """调用OpenAI兼容的Chat Completions接口"""
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个专业的数学教学助手。"},
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            return response.choices[0].message.content
        except ImportError:
            raise ImportError("需要安装openai: pip install openai")
        except Exception as e:
            return f"[LLM调用失败: {str(e)}]"


class SiliconFlowClient(OpenAIClient):
    """SiliconFlow DeepSeek 客户端（OpenAI 兼容协议）"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        api_key = api_key or os.getenv("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError("需要设置SILICONFLOW_API_KEY环境变量或传入api_key")
        resolved_model = model or os.getenv("SILICONFLOW_MODEL")
        resolved_base_url = base_url or os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
        super().__init__(api_key=api_key, model=resolved_model, base_url=resolved_base_url)


class MockLLMClient(LLMClient):
    """模拟LLM客户端（用于测试）"""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """返回模拟响应"""
        return f"""
[模拟LLM响应]
根据提示词：{prompt[:100]}...

这是一个模拟的LLM响应。请配置真实的LLM客户端（如OpenAI）以获得实际功能。
"""


# 默认使用Mock客户端，优先读取 SiliconFlow，其次 OpenAI
def get_default_llm_client() -> LLMClient:
    """获取默认LLM客户端"""
    sf_api_key = os.getenv("SILICONFLOW_API_KEY")
    if sf_api_key:
        return SiliconFlowClient(api_key=sf_api_key)
    
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAIClient(api_key=api_key)
    
    return MockLLMClient()

