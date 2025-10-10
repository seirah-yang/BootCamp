{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyOd/hZ3yhxZcUvEzeHx7C/9",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/seirah-yang/BootCamp/blob/main/LLM_%EA%B8%B0%EB%B0%98%EC%9D%98_%EB%A6%AC%EB%9E%AD%ED%82%B9.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "<기본 RAG> - 벡터 검색으로 문서 4개를 찾아옵니다\n",
        "- LLM 호출 1회: 검색된 4개 문서 전체를 바탕으로 답변을 생성합니다\n",
        "- 총 LLM 호출: 1회\n",
        "\n",
        "<리랭킹 적용 RAG> - 벡터 검색으로 문서 4개를 찾아옵니다\n",
        "- LLM 호출 4회: 각 문서와 질문의 관련성을 1-10점 사이로 평가합니다\n",
        "- LLM 호출 1회: 관련성 점수가 높은 상위 2개 문서만으로 답변을 생성합니다\n",
        "- 총 LLM 호출: 5회"
      ],
      "metadata": {
        "id": "PD6o_LjqUMsk"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import os\n",
        "import urllib.request\n",
        "import json\n",
        "from typing import List\n",
        "from langchain_openai import ChatOpenAI, OpenAIEmbeddings\n",
        "from langchain_community.document_loaders import PyPDFLoader\n",
        "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
        "from langchain_chroma import Chroma\n",
        "from langchain.schema import Document\n",
        "import requests"
      ],
      "metadata": {
        "id": "Qkz6JQbxUcev"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 분석할 PDF 파일을 웹에서 다운로드.\n",
        "url = \"https://github.com/llama-index-tutorial/llama-index-tutorial/raw/main/ch07/2023_%EB%B6%81%ED%95%9C%EC%9D%B8%EA%B6%8C%EB%B3%B4%EA%B3%A0%EC%84%9C.pdf\"\n",
        "filename = \"2023_북한인권보고서.pdf\"\n",
        "\n",
        "response = requests.get(url)\n",
        "with open(filename, \"wb\") as f:\n",
        "    f.write(response.content)\n",
        "\n",
        "print(f\"{filename} 다운로드 완료\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "S2Y43XOGUPdN",
        "outputId": "55ccdb86-b77c-47fc-9b99-33b14a6c1246"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2023_북한인권보고서.pdf 다운로드 완료\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import torch\n",
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n",
        "\n",
        "model_name = \"skt/A.X-4.0\"\n",
        "model = AutoModelForCausalLM.from_pretrained(\n",
        "    model_name,\n",
        "    torch_dtype=torch.bfloat16,\n",
        "    device_map=\"auto\",\n",
        ")\n",
        "model.eval()\n",
        "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
        "\n",
        "messages = [\n",
        "    {\"role\": \"system\", \"content\": \"당신은 사용자가 제공하는 영어 문장들을 한국어로 번역하는 AI 전문가입니다.\"},\n",
        "    {\"role\": \"user\", \"content\": \"The first human went into space and orbited the Earth on April 12, 1961.\"},\n",
        "]\n",
        "input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors=\"pt\").to(model.device)\n",
        "\n",
        "with torch.no_grad():\n",
        "    output = model.generate(\n",
        "        input_ids,\n",
        "        max_new_tokens=128,\n",
        "        do_sample=False,\n",
        "    )\n",
        "\n",
        "len_input_prompt = len(input_ids[0])\n",
        "response = tokenizer.decode(output[0][len_input_prompt:], skip_special_tokens=True)\n",
        "print(response)"
      ],
      "metadata": {
        "id": "lsPI1BKwWPnX"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from openai import OpenAI\n",
        "def call(messages, model):\n",
        "    completion = client.chat.completions.create(\n",
        "        model=model,\n",
        "        messages=messages,\n",
        "    )\n",
        "    print(completion.choices[0].message)\n",
        "\n",
        "client = OpenAI(\n",
        "    base_url=\"http://localhost:8000/v1\",\n",
        "    api_key=\"api_key\"\n",
        ")\n",
        "model = \"skt/A.X-4.0\"\n",
        "messages = [{\"role\": \"user\", \"content\": \"content\"}]\n",
        "call(messages, model)\n",
        "embed_model = OpenAIEmbeddings(model=\"text-embedding-3-large\")  # 임베딩 모델 사용\n",
        "\n",
        "# 문서 분할 설정\n",
        "text_splitter = RecursiveCharacterTextSplitter(\n",
        "    chunk_size=300,\n",
        "    chunk_overlap=50,\n",
        ")\n",
        "\n",
        "# PDF 문서를 읽고 벡터 인덱스 생성\n",
        "loader = PyPDFLoader(\"2023_북한인권보고서.pdf\")  # PDF 문서 로더\n",
        "documents = loader.load()  # 문서에서 텍스트 추출\n",
        "chunks = text_splitter.split_documents(documents)  # 문서 분할\n",
        "vector_store = Chroma.from_documents(chunks, embed_model)  # 추출된 텍스트로 벡터 인덱스 생성\n"
      ],
      "metadata": {
        "id": "x1BrIwaNVRNl"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class DocumentScorer:\n",
        "    # LLM을 사용해 문서의 관련성을 정밀하게 평가하고 점수를 매기는 클래스\n",
        "\n",
        "    def __init__(self, llm):\n",
        "        self.llm = llm\n",
        "\n",
        "    def evaluate_document(self, query: str, content: str) -> float:\n",
        "        # LLM을 사용해 문서와 쿼리 간의 의미적 관련성을 1-10점으로 평가\n",
        "        prompt = f\"\"\"\n",
        "        아래 주어진 질문과 문서의 관련성을 평가해주세요.\n",
        "\n",
        "        [평가 기준]\n",
        "        - 문서가 질문에서 요구하는 정보를 직접적으로 포함하면 8-10점\n",
        "        - 문서가 질문과 관련된 맥락을 포함하지만 직접적인 답이 아니면 4-7점\n",
        "        - 문서가 질문과 거의 관련이 없으면 1-3점\n",
        "\n",
        "        [주의사항]\n",
        "        - 단순히 비슷한 단어가 등장하는 것은 높은 점수의 근거가 될 수 없습니다\n",
        "        - 질문의 의도와 문맥을 정확히 파악하여 평가해주세요\n",
        "        - 시간, 장소, 수치 등 구체적인 정보의 일치 여부를 중요하게 고려해주세요\n",
        "\n",
        "        질문: {query}\n",
        "        문서: {content}\n",
        "\n",
        "        응답은 반드시 다음 JSON 형식이어야 합니다. 백틱은 쓰지마십시오.:\n",
        "        {{\"relevance_score\": float}}\n",
        "        \"\"\"\n",
        "\n",
        "        try:\n",
        "            # LLM에 프롬프트를 전송하고 JSON 형식의 응답을 받음\n",
        "            response = self.llm.invoke(prompt)\n",
        "            # 응답에서 relevance_score 값을 추출\n",
        "            score = json.loads(response.content)[\"relevance_score\"]\n",
        "            # 점수를 float로 변환하여 반환\n",
        "            return float(score)\n",
        "        except Exception as e:\n",
        "            print(f\"Error occurred: {str(e)}\")\n",
        "            return 5.0  # 에러 발생시 중간 점수로 처리하여 시스템 안정성 유지\n",
        "\n",
        "    def postprocess_documents(self, documents: List[Document], query: str) -> List[Document]:\n",
        "        # 벡터 검색으로 찾은 4개 문서를 LLM으로 재평가하여 최적의 2개 선택\n",
        "        print('\\n=== LLM이 4개의 검색 결과에 대해서 관련성을 평가합니다. ===')\n",
        "        scored_docs = []\n",
        "        for doc in documents:\n",
        "            # 현재 처리 중인 문서에서 순수 텍스트 컨텐츠만 추출\n",
        "            content = doc.page_content\n",
        "            # LLM으로 문서 관련성 점수 계산 (1-10 사이 점수)\n",
        "            score = self.evaluate_document(query, content)\n",
        "            # 디버깅/모니터링을 위해 각 문서의 내용과 점수를 출력\n",
        "            print(f\"\\nLLM 기반의 평가:\\n{content}\\n=> 점수: {score}\\n\")\n",
        "            # 현재 문서와 계산된 점수를 튜플로 저장\n",
        "            scored_docs.append((doc, score))\n",
        "\n",
        "        # 모든 문서를 점수 기준 내림차순으로 정렬하고 상위 2개만 선택하여 반환\n",
        "        ranked_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)\n",
        "        return [doc for doc, _ in ranked_docs[:2]]\n"
      ],
      "metadata": {
        "id": "YBT3YfLmViq6"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "class SemanticRanker:\n",
        "    # 벡터 검색 결과에 LLM 기반 의미적 평가를 적용하여 최적의 문서를 선별하는 시스템\n",
        "\n",
        "    def __init__(self, vector_store, scorer):\n",
        "        # 생성자에서 벡터 검색용 저장소와 LLM 기반 문서 평가기 인스턴스를 받아 저장\n",
        "        self.vector_store = vector_store  # 벡터 검색용 저장소\n",
        "        self.scorer = scorer  # LLM 기반 문서 평가기\n",
        "\n",
        "    def retrieve(self, query: str) -> List[Document]:\n",
        "        # 벡터 검색으로 유사도 기반 후보 문서 4개를 추출하고 LLM으로 재평가\n",
        "        vector_results = self.vector_store.similarity_search(query, k=4)\n",
        "\n",
        "        # 초기 벡터 검색 결과를 디버깅/분석용으로 출력\n",
        "        print(\"\\n=== 실제 검색 결과 (Top 4) ===\")\n",
        "        for i, doc in enumerate(vector_results, 1):\n",
        "            print(f\"\\n검색 문서 {i}:\")\n",
        "            print(doc.page_content)\n",
        "\n",
        "        # LLM으로 문서들을 재평가하고 재정렬하여 최적의 2개 선택\n",
        "        reranked_results = self.scorer.postprocess_documents(vector_results, query)\n",
        "\n",
        "        # 최종 선별된 문서를 디버깅/분석용으로 출력\n",
        "        print(\"\\n=== LLM의 리랭킹 결과 (Top 2) ===\")\n",
        "        for i, doc in enumerate(reranked_results, 1):\n",
        "            print(f\"\\n검색 문서 {i}:\")\n",
        "            print(doc.page_content)\n",
        "\n",
        "        return reranked_results"
      ],
      "metadata": {
        "id": "i0N-wZiKVulg"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# 문서 평가 및 검색 시스템 선언(초기화)\n",
        "scorer = DocumentScorer(llm)  # LLM 기반 문서 평가기 생성\n",
        "ranker = SemanticRanker(vector_store, scorer)  # 벡터 검색과 LLM 평가를 결합한 시스템 생성"
      ],
      "metadata": {
        "id": "V9JM_I8eWCrk"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}