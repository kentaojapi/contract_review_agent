import json
import os
from functools import cached_property

from dotenv import load_dotenv
from openai import OpenAI
from swarm import Swarm, Agent
from tavily import TavilyClient


load_dotenv()


class SwarmClient:
    def __init__(self) -> None:
        self.openai_api_key = os.environ['OPENAI_API_KEY']

    @cached_property
    def client(self):
        sub_client = OpenAI(api_key=self.openai_api_key)
        return Swarm(client=sub_client)


class Tools:
    def __init__(self) -> None:
        self.tavily_client = TavilyClient(api_key=os.environ['TAVILY_API_KEY'])

    def summarize_contract(self, context_variables) -> str:
        """
        summarize_contractは契約書の要約を生成するために使用されるツール
        """
        customer_document = context_variables.get("customer_document", None)
        prompt = "契約書の要約を行ってください。"
        prompt += f"アップロードされた契約書の内容は以下です:"
        prompt += customer_document
        return prompt

    def revise_contract(self, context_variables) -> str:
        """
        revise_contractは契約書の修正提案を生成するために使用されるツール
        """
        customer_document = context_variables.get("customer_document", None)
        prompt = "契約書の修正提案を行ってください。"
        prompt += f"アップロードされた契約書の内容は以下です:"
        prompt += customer_document
        return prompt

    def search_web(self, search_query: str) -> str:
        """
        Web検索を実行するツール
        """
        result = self.tavily_client.qna_search(query=search_query)
        return result

class Agents:
    def __init__(self) -> None:
        self.model = os.environ['LLM_MODEL']
        self.tools = Tools()

    def triage(self) -> Agent:        
        return Agent(
            name="Triage Agent",
            model=self.model,
            instructions = self._triage_instrunction,
            functions = [
                self.summarizer,
                self.reviser,
            ],
        )

    def _triage_instrunction(self, context_variables) -> str:
        customer_document = context_variables.get("customer_document", None)
        prompt = f"ユーザは契約書をアップロードし、その要約もしくは修正をリクエストします。"
        prompt += "あなたはユーザーのリクエストをトリアージし、適切なエージェントに転送するためのツールを呼び出します。"
        prompt += "適切なエージェントに転送する準備ができたら、直接ツールを呼び出してください。"
        prompt += "エージェントにリクエストをトリアージするために追加情報が必要な場合、説明せずに直接質問をしてください。"
        prompt += f"アップロードされた契約書の内容はここにあります: {customer_document}"
        return prompt

    def summarizer(self) -> Agent:
        return Agent(
            name="Generate Summary Agent",
            model=self.model,
            instruction=self._summary_instruction,    
            functions=[self.tools.summarize_contract, self.triage],
        )

    def _summary_instruction(self) -> str:
        prompt = "あなたは契約書の要約を生成するエージェントです。"
        prompt += "契約書の要約を生成するために、summarize_contractというツールを使用して要約を生成し、出力してください."
        prompt += "要約を生成する必要がないと判断した場合は、triage agentに転送してください。"
        return prompt

    def reviser(self) -> Agent:
        return Agent(
            name="Generate Revised Draft Agent",
            model=self.model,
            instruction=self._revise_instruction, 
            functions=[self.tools.revise_contract, self.triage],
        )

    def _revise_instruction(self) -> str:
        prompt = "あなたは契約書の修正提案を生成するエージェントです。"
        prompt += "契約書の修正提案を生成するために、revise_contractというツールを使用して修正提案を生成し出力してください。"
        prompt += "修正提案を生成する必要がないと判断した場合は、triage agentに転送してください。"
        return prompt

    def websearcher(self) -> Agent:
        return Agent(
            name="Web Search Agent",
            model=self.model,
            instructions=self._websearch_instruction,
            functions=[self.tools.search_web, self.reviser],
        )

    def _websearch_instruction(self, context_variables) -> str:
        customer_document = context_variables.get("customer_document", None)
        prompt = "ユーザーの質問に基づいてWeb検索を行い、その結果を提供してください。"
        prompt += "その後、revise agentに転送してください。"
        return prompt


if __name__ == '__main__':
    messages = [{"role":"user", 'content': "最新の器物損壊罪の内容をWebで検索した上で、契約書を修正してください。"}]
    context_variables = {
        "customer_document": """
        甲及び乙は、故意もしくは過失により、又は本契約に違反した場合、相手方が被った損害（通常損害及び特別損害）を賠償する。
        """
    }

    agents = Agents()
    client = SwarmClient().client
    response = client.run(
        agent=agents.triage(),
        messages=messages,
        context_variables=context_variables,
        debug=True
    )

    print(response.messages[-1]['content'])
