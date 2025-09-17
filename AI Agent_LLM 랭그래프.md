# AI Agent_LLM

# 부제 : LangGraph_simple

---

## LangGraph

**= 자연어 처리와 AI 응용 프로그램 개발을 위한 강력한 프레임워크**

복잡한 언어 모델과의 상호작용을 효율적이고 구조화된 방식으로 구현할 수 있도록 한다. 

다양한 데이터 소스와 언어 모델을 통합하여 지능형 응답 생성, 정보 검색, 텍스트 분석 등의 고도화된 시스템을 구축할 수 있다.

![스크린샷 2025-09-17 오후 9.46.38.png](AI%20Agent_LLM%20271d44a085da80d9bdddd9591d739952/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2025-09-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_9.46.38.png)

---

### 1. 주요 목적:

- LangGraph는 복잡한 워크플로우와 정교한 의사결정 프로세스를 구현하는 데 특화되어 있다. 여러 단계의 처리 과정이나 조건부 로직이 필요한 고급 AI 시스템에 적합
- LangChain은 LLM을 다양한 외부 도구와 쉽게 통합하고, 간단한 체인 구조로 애플리케이션을 구성하는 데 중점을 둔다. 빠른 프로토타이핑과 기본적인 LLM 기반 애플리케이션 개발에 유용

### 2. 구조:

- LangGraph는 그래프 기반 구조를 채택하여 노드와 엣지로 구성된 유연한 워크플로우를 만들 수 있다. 이는 복잡한 로직과 다단계 프로세스를 직관적으로 모델링하는 데 도움이 된다.
- LangChain은 체인과 에이전트 기반 구조를 사용한다. 이는 선형적인 처리 과정이나 미리 정의된 에이전트 패턴을 구현하기 쉽게 만든다.

### 3. 상태 관리:

- LangGraph는 명시적이고 세밀한 상태 관리를 제공한다. 개발자가 각 단계에서 상태를 직접 제어하고 수정할 수 있어, 복잡한 상태 변화를 정확하게 추적하고 관리할 수 있다.
- LangChain은 상대적으로 암시적이고 자동화된 상태 관리를 제공한다. 이는 개발 과정을 단순화하지만, 세부적인 상태 제어가 필요한 경우 제한적일 수 있다.

### 4. 유연성:

- LangGraph는 높은 유연성을 제공하여 커스텀 로직을 쉽게 구현할 수 있다. 개발자가 원하는 대로 그래프 구조를 설계하고 각 노드의 동작을 상세하게 정의할 수 있다.
- LangChain은 미리 정의된 컴포넌트를 중심으로 구성되어 있어, 기본적인 기능을 빠르게 구현할 수 있지만 고도로 커스터마이즈된 로직을 구현하는 데는 상대적으로 제한이 있을 수 있다.

### 5. 학습 곡선:

- LangGraph는 그래프 이론과 상태 관리에 대한 이해가 필요하여 상대적으로 가파른 러닝 커브를 갖는다. 하지만 이를 통해 더 복잡하고 정교한 시스템을 구축할 수 있다.
- LangChain은 직관적인 API와 풍부한 예제로 인해 상대적으로 완만한 러닝 커브를 갖는다. 초보자도 빠르게 기본적인 LLM 애플리케이션을 만들 수 있다.

### 6. 용도:

- LangGraph는 복잡한 AI 시스템이나 다중 에이전트 시스템을 구축하는 데 적합
- 여러 AI 모델이 상호작용하거나 복잡한 의사결정 과정이 필요한 프로젝트에 이상적
- LangChain은 간단한 LLM 애플리케이션이나 RAG(Retrieval-Augmented Generation) 시스템을 구축하는 데 주로 사용
- 빠른 개발과 프로토타이핑이 필요한 프로젝트에 적합

**결론:** 프로젝트의 복잡성, 필요한 유연성, 개발 팀의 경험 수준, 그리고 구현하고자 하는 기능에 따라 LangGraph와 LangChain 중 적절한 도구를 선택한다. 

✅ 복잡하고 맞춤화된 AI 시스템이 필요하다면 **LangGraph**

✅ 빠른 개발과 간단한 LLM 통합이 목표라면 **LangChain**

                                                            자료출처: https://wikidocs.net/261585

---

## RAG

자료출처: https://aws.amazon.com/ko/what-is/retrieval-augmented-generation/

## **검색 증강 생성이란?**

검색 증강 생성(RAG)은 대규모 언어 모델의 출력을 최적화하여 응답을 생성하기 전에 훈련 데이터 소스 외부의 신뢰할 수 있는 기술 자료를 참조하도록 하는 프로세스입니다. 대규모 언어 모델(LLM)은 방대한 양의 데이터를 기반으로 훈련되며 수십억 개의 파라미터를 사용하여 질문에 대한 답변, 언어 번역, 문장 완성 등의 작업에서 독창적인 결과를 생성합니다. RAG는 이미 강력한 LLM의 기능을 특정 도메인이나 조직의 내부 기술 자료로 확장하므로 모델을 다시 훈련할 필요가 없습니다. 이는 LLM 결과를 개선하여 다양한 상황에서 연관성, 정확성 및 유용성을 유지하기 위한 비용 효율적인 접근 방식입니다.

## **검색 증강 생성이 중요한 이유는 무엇인가요?**

LLM은 지능형 [챗봇](https://aws.amazon.com/what-is/chatbot/) 및 기타 [자연어 처리(NLP)](https://aws.amazon.com/what-is/nlp/) 애플리케이션을 지원하는 핵심 [인공 지능(AI)](https://aws.amazon.com/what-is/artificial-intelligence/) 기술입니다. 신뢰할 수 있는 지식 소스를 상호 참조하여 다양한 상황에서 사용자 질문에 답변할 수 있는 봇을 만드는 것이 LLM의 목표입니다. 안타깝게도 LLM 기술의 특성상 LLM 응답에 대한 예측이 불가능합니다. 또한 LLM 훈련 데이터는 정적이며 보유한 지식은 일정 기간 동안만 유용합니다.

LLM의 알려진 문제점은 다음과 같습니다.

- **답이 없을 때 허위 정보를 제공**합니다.
- 사용자가 구체적이고 **최신의 응답을 기대할 때 오래되었거나 일반적인 정보를 제공**합니다.
- **신뢰할 수 없는 출처로부터 응답을 생성**합니다.
- **다양한 훈련 소스에서 동일한 용어를 사용하여 다른 내용을 설명하면서 용어 혼동으로 인해 응답이 부정확**합니다.

[대형 언어 모델](https://aws.amazon.com/what-is/large-language-model/)은 현재 상황에 대한 최신 정보는 없지만 항상 절대적인 자신감을 가지고 모든 질문에 답변하는 열정적인 신입 사원으로 생각할 수 있습니다. 안타깝게도 이러한 태도는 사용자 신뢰에 부정적인 영향을 미칠 수 있으며 챗봇이 모방해서는 안 되는 것입니다.

RAG는 이러한 문제 중 일부를 해결하기 위한 접근 방식입니다. LLM을 리디렉션하여 신뢰할 수 있는 사전 결정된 지식 출처에서 관련 정보를 검색합니다. 조직은 생성된 텍스트 출력을 더 잘 제어할 수 있으며 사용자는 LLM이 응답을 생성하는 방식에 대한 인사이트를 얻을 수 있습니다.

## **검색 증강 생성의 이점은 무엇인가요?**

RAG 기술은 조직의 [생성형 AI](https://aws.amazon.com/what-is/generative-ai/) 관련 작업에 있어 여러 가지 이점을 제공합니다.

### **비용 효율적인 구현**

챗봇 개발은 일반적으로 [파운데이션 모델](https://aws.amazon.com/what-is/foundation-models/)을 사용하여 시작됩니다. 파운데이션 모델(FM)은 광범위한 일반화 데이터와 레이블이 지정되지 않은 데이터에 대해 훈련된 API 액세스 가능 LLM입니다. 조직 또는 도메인별 정보를 위해 FM을 재교육하는 데 컴퓨팅 및 재정적 비용이 많이 소요됩니다. RAG를 통해 LLM에 새 데이터를 더 비용 효율적으로 도입할 수 있습니다. 그리고 생성형 인공 지능(생성형 AI) 기술을 보다 폭넓게 접근하고 사용할 수도 있습니다.

### **최신 정보**

LLM의 원본 훈련 데이터 소스가 요구 사항에 적합하더라도 연관성을 유지하기가 어렵습니다. 개발자는 RAG를 사용하여 생성형 모델에 최신 연구, 통계 또는 뉴스를 제공할 수 있습니다. RAG를 사용하여 LLM을 라이브 소셜 미디어 피드, 뉴스 사이트 또는 기타 자주 업데이트되는 정보 소스에 직접 연결할 수 있습니다. 그러면 LLM은 사용자에게 최신 정보를 제공할 수 있습니다.

### **사용자 신뢰 강화**

RAG은 LLM이 소스의 저작자 표시를 통해 **정확한 정보를 제공**할 수 있게 합니다. 출력에는 소스에 대한 인용 또는 참조가 포함될 수 있습니다. 추가 설명이나 세부 정보가 필요한 경우 사용자가 소스 문서를 직접 찾아볼 수도 있습니다. 이를 통해 생성형 AI 솔루션에 대한 신뢰와 확신을 높일 수 있습니다.

### **개발자 제어 강화**

개발자는 RAG를 사용하여 채팅 애플리케이션을 보다 효율적으로 테스트하고 개선할 수 있습니다. 변화하는 요구 사항 또는 부서 간 사용에 맞게 LLM의 정보 소스를 제어하고 변경할 수 있습니다. 또한 개발자는 민감한 정보 검색을 다양한 인증 수준으로 제한하여 **LLM이 적절한 응답 생성을 유도할 수 있**습니다. 또한 LLM이 특정 질문에 대해 잘못된 정보 소스를 참조하는 경우 문제를 해결하고 수정할 수도 있습니다. 조직은 더 광범위한 애플리케이션을 대상으로 생성형 AI 기술을 보다 자신 있게 구현할 수 있습니다.

## **검색 증강 생성은 어떻게 작동하나요?**

RAG가 없는 경우 LLM은 사용자 입력을 바탕으로 훈련한 정보 또는 이미 알고 있는 정보를 기반으로 응답을 생성합니다. RAG에는 사용자 입력을 활용하여 먼저 새 데이터 소스에서 정보를 가져오는 정보 검색 구성 요소가 도입되었습니다. 사용자 **쿼리와 관련 정보가 모두 LLM에 제공**됩니다. **LLM은 새로운 지식과 학습 데이터를 사용하여 더 나은 응답을 생성**합니다. 프로세스의 개요를 아래에서 확인하세요.

### **외부 데이터 생성**

LLM의 원래 학습 데이터 세트 외부에 있는 새 데이터를 ***외부 데이터***라고 합니다. API, 데이터베이스 또는 문서 리포지토리와 같은 여러 데이터 소스에서 가져올 수 있습니다. 데이터의 형식은 파일, **데이터베이스 레코드 또는 긴 형식의 텍스트**와 같이 다양합니다. *임베딩 언어 모델*이라고 하는 또 다른 AI 기법은 데이터를 수치로 변환하고 벡터 데이터베이스에 저장합니다. 이 프로세스는 생성형 AI 모델이 이해할 수 있는 지식 라이브러리를 생성합니다.

### **관련 정보 검색**

연관성 검색을 수행하는 단계는 다음과 같습니다. **사용자 쿼리는 벡터 표현으로 변환되고 벡터 데이터베이스와 매칭**됩니다. 예를 들어 조직의 인사 관련 질문에 답변할 수 있는 스마트 챗봇을 생각할 수 있습니다. 직원이 *“연차휴가는 얼마나 남았나요?“*라고 검색하면 시스템은 개별 직원의 과거 휴가 기록과 함께 연차 휴가 정책 문서를 검색합니다. 이러한 특정 문서는 직원이 입력한 내용과 관련이 높기에 반환됩니다. 수학적 벡터 계산 및 표현을 사용하여 연관성이 계산 및 설정됩니다.

### **LLM 프롬프트 확장**

다음으로 RAG 모델은 검색된 관련 데이터를 컨텍스트에 추가하여 사용자 입력(또는 프롬프트)을 보강합니다. 이 단계에서는 프롬프트 엔지니어링 기술을 사용하여 LLM과 효과적으로 통신합니다. 확장된 프롬프트를 사용하면 대규모 언어 모델이 사용자 쿼리에 대한 정확한 답변을 생성할 수 있습니다.

### **외부 데이터 업데이트**

만약에 외부 데이터가 시간이 경과된 데이터가 된다면 어떻게 될까요? 최신 정보 검색을 유지하기 위해 문서를 비동기적으로 업데이트하고 문서의 임베딩 표현을 업데이트합니다. 자동화된 실시간 프로세스 또는 주기적 배치 처리를 통해 이 작업을 수행할 수 있습니다. 변경 관리에 다양한 데이터 과학 접근 방식을 사용할 수 있기 때문에 데이터 분석에서 흔히 발생하는 과제이기도 합니다.

![                              이미지 출처: AWS [https://aws.amazon.com/ko/what-is/retrieval-augmented-generation/](https://aws.amazon.com/ko/what-is/retrieval-augmented-generation/)](AI%20Agent_LLM%20271d44a085da80d9bdddd9591d739952/image.png)

                              이미지 출처: AWS [https://aws.amazon.com/ko/what-is/retrieval-augmented-generation/](https://aws.amazon.com/ko/what-is/retrieval-augmented-generation/)

## AI Agent 실습

```jsx
!pip install langchain langgraph langchain-community chromadb sqlite-utils

from langchain_community.llms import Ollama  # Ollama LLM 사용
from langchain_core.prompts import PromptTemplate  # 프롬프트 템플릿
from langgraph.graph import StateGraph, END  # LangGraph 상태 머신
from typing import TypedDict  # 타입 정의용

# 1. 상태 정의
class AgentState(TypedDict):  # 상태 타입 정의
    query: str  # 사용자 질의
    symptoms: str  # 추출된 증상
    disease_candidates: str  # 질병 후보
    result: str  # 최종 응답

# 2. LLM 초기화
llm = Ollama(model="exaone3.5:7.8b")  # Ollama 모델 로딩

# 3. 에이전트 정의
extractor_prompt = PromptTemplate.from_template("""
                                                사용자의 질문에서 증상에 해당하는 단어 또는 구를 추출하세요.
                                                결과는 쉼표로 구분된 문자열로 출력하세요.
                                                질문: {query}
                                                """)  # 증상 추출 프롬프트

def extractor_agent(state: AgentState):  # 증상 추출 함수
    chain = extractor_prompt | llm  # 프롬프트 체인
    symptoms = chain.invoke({"query": state["query"]})  # LLM 실행
    return {**state, "symptoms": symptoms.strip()}  # 상태에 추가

matcher_prompt = PromptTemplate.from_template("""
                                                다음 증상 목록을 바탕으로 가장 가능성 높은 질병 이름 3개를 쉼표로 추정하세요.
                                                증상: {symptoms}
                                                """)  # 질병 후보 추정 프롬프트

def matcher_agent(state: AgentState):  # 질병 후보 추정
    chain = matcher_prompt | llm
    candidates = chain.invoke({"symptoms": state["symptoms"]})
    return {**state, "disease_candidates": candidates.strip()}

answer_prompt = PromptTemplate.from_template("""
                                            사용자의 증상은 다음과 같습니다: {symptoms}

                                            예측된 질병 후보: {disease_candidates}

                                            위 내용을 바탕으로 사용자에게 알기 쉽게 설명해주세요.
                                            """)  # 최종 응답 생성 프롬프트

def answer_agent(state: AgentState):  # 응답 생성 에이전트
    chain = answer_prompt | llm  # 프롬프트와 LLM을 연결하여 실행 체인 구성
    answer = chain.invoke({
        "symptoms": state["symptoms"],
        "disease_candidates": state["disease_candidates"]
    })
    return {**state, "result": answer.strip()}

# 4. LangGraph 정의
from langgraph.graph import StateGraph  # LangGraph 구성 요소

graph = StateGraph(AgentState)  # 그래프 정의
graph.add_node("extractor", extractor_agent)  # 노드 추가
graph.add_node("matcher", matcher_agent)
graph.add_node("answer", answer_agent)

graph.set_entry_point("extractor")  # 시작 노드 설정
graph.add_edge("extractor", "matcher")  # 노드 간 연결 정의
graph.add_edge("matcher", "answer")
graph.add_edge("answer", END)  # 종료 노드 설정

app = graph.compile()  # 그래프 컴파일

print("============================== LangGraph 구조:")
app.get_graph().print_ascii()  # 구조 출력

# 5. 실행 예시
query = "기침이 심하고 목이 아프고 열이 납니다"  # 사용자 질문
result = app.invoke({"query": query})  # 실행

print("============================== 최종 응답:")
print(result["result"])  

		**# 결과출력** 
		**/tmp/ipykernel_942723/1617558422.py:14: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.
		  llm = Ollama(model="exaone3.5:7.8b")  # Ollama 모델 로딩
		============================== LangGraph 구조:**
		**+-----------+  
		| __start__ |  
		+-----------+  
		      *        
		      *        
		      *        
		+-----------+  
		| extractor |  
		+-----------+  
		      *        
		      *        
		      *        
		 +---------+   
		 | matcher |   
		 +---------+   
		      *        
		      *        
		      *        
		  +--------+   
		  | answer |   
		  +--------+   
		      *        
		      *        
		      *        
		 +---------+   
		 | __end__ |   
		 +---------+   
		============================== 최종 응답:**
안녕하세요. 제공해주신 증상 (기침, 목 아픔, 열)을 바탕으로 몇 가지 가능한 질병에 대해 간단히 설명해 드리겠습니다:

1. **감기**:
   - **특징**: 감기는 가장 흔하게 겪는 호흡기 질환 중 하나입니다. 주로 콧물, 코막힘, 기침, 목 아픔, 가벼운 발열을 동반합니다. 증상은 대체로 경미하고 며칠 내에 호전되는 경향이 있습니다.
   - **예후**: 대부분의 경우 자가 치유가 가능하며, 충분한 휴식과 수분 섭취가 도움이 됩니다.

2. **독감 (인플루엔자)**:
   - **특징**: 독감은 감기보다 더 심한 증상을 동반합니다. 높은 발열, 심한 기침, 근육통, 두통, 피로감 등이 특징입니다. 감기와 비슷한 증상이 있지만, 증상의 강도가 더 높을 수 있습니다.
   - **예후**: 빠른 치료와 휴식이 중요하며, 경우에 따라 항바이러스 약물이 처방될 수 있습니다.

3. **코로나19**:
   - **특징**: 코로나19는 기침, 목 아픔, 발열 외에도 호흡 곤란, 피로, 미각/후각 상실 등 다양한 증상을 동반할 수 있습니다. 초기 증상이 감기나 독감과 유사할 수 있습니다.
   - **예후**: 감염의 정도에 따라 다르지만, 중증 경우에는 입원 치료가 필요할 수 있습니다. 예방 접종과 함께 건강 관리가 중요합니다.

### 조언:
- **증상 확인**: 위 증상들이 지속되거나 심해진다면, 의료 전문가와 상담하는 것이 가장 안전합니다. 특히 코로나19 검사나 독감 검사를 받아보는 것을 고려해 보세요.
- **자기 관리**: 충분한 휴식, 수분 섭취, 적절한 약물 복용(의사의 지시에 따라)으로 증상 완화에 도움이 될 수 있습니다.
- **예방**: 손 씻기, 마스크 착용, 사회적 거리 두기 등 기본적인 예방 조치를 지키는 것이 중요합니다.

증상이 지속되거나 악화되면 즉시 의료 기관에 연락하시거나 방문하시기 바랍니다. 건강하시길 바랍니다!

```

## LangGraph_실습하기

### 서울명서 스케줄 짜기

```python
!pip install colab-xterm

%load_ext colabxterm

%xterm

# 아래를 한줄씩 마우스 오른쪽 붙여넣기로
# curl -fsSL https://ollama.com/install.sh | sh
# ollama serve & ollama pull exaone3.5:2.4b

!pip install langchain langgraph langchain-community chromadb sqlite-utils

!pip install grandalf

## Mission  II
@ 랭그래프 마스터하기
1. 에이전트 2개
2. 주제: 서울명소 도출 -> 여행 스케줄 짜기

from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    query: str
    place: str
    result: str

llm = Ollama(model="exaone3.5:2.4b")

matcher_prompt = PromptTemplate.from_template("""
                                                세계 각국에서 소개하는 서울에서 꼭 가봐야 하는 장소 top 10을 모아주세요.
                                                그 중에서 가장 많이 언급되는 명소 3위와 키워드를 제시해 주세요.
                                                명소와 관련 기사와 정부기관 홍보영상, 근거가 명확한 자료를 기반으로
                                                명소 관련 키워드, 한국 경제 시장, 전통 먹거리 시장, 요즘 먹거리 트렌드, 쇼핑 트렌드, 판매하는 물건, 기념품,
                                                명소와 관련된 역사 등 각 영역에서 베스트 후보 5개 이상을 선정해주세요
                                                정보: {query}
                                                """)

def matcher_agent(state: AgentState):
    print("A")
    chain = matcher_prompt | llm
    place = chain.invoke({"query": state["query"]})
    return {**state, "place": place.strip()}

answer_prompt = PromptTemplate.from_template("""
                                                가장 많이 언급되는 명소 3위와 각 영역에서 베스트 후보 5개 이상을 선정한 후,
                                                키워드를 각 문장에 포함하여 장소, 이용가능시간, 특징, 주의사항 등 세부내용을 이해하기 쉽게 설명하세요.
                                                서울지역에 사는 사람이 아닌, 타 지역에서 방문하는 여행자를 위한 1지역당 반나절 여행 스케줄을 짜세요.
                                                하루에 3군데 돌아다닐 수 있는 수준으로 제시하세요.
                                                비가 올 것을 대비하여 대안 동선/실내 장소 및 대비 방안도 포함하세요.
                                                이용 가능한 대중교통 내용과 이용방법 상세를 포함하세요.
                                                명소 : {place}
                                                """)

def answer_agent(state: AgentState):
    print("B")
    chain = answer_prompt | llm
    answer = chain.invoke({
        "place": state["place"]
    })
    return {**state, "result": answer.strip()}
    
# 4. LangGraph 정의
from langgraph.graph import StateGraph  # LangGraph 구성 요소

graph = StateGraph(AgentState)
graph.add_node("matcher", matcher_agent)
graph.add_node("answer", answer_agent)

graph.set_entry_point("matcher")
graph.add_edge("matcher", "answer")
graph.add_edge("answer", END)

app = graph.compile()

print("============================== LangGraph 구조:")
app.get_graph().print_ascii()

# 5. 실행 예시
query = "서울에 요즘 사람들은 어디를 많이 고려해서 3박4일 일정인데 어디에 갈지 스케줄 짜줘"
result = app.invoke({"query": query})

print("============================== 최종 응답:")
print(result["result"])

**# 결과 출력 
============================== LangGraph 구조:
+-----------+  
| __start__ |  
+-----------+  
      *        
      *        
      *        
 +---------+   
 | matcher |   
 +---------+   
      *        
      *        
      *        
  +--------+   
  | answer |   
  +--------+   
      *        
      *        
      *        
 +---------+   
 | __end__ |   
 +---------+   
A
B
============================== 최종 응답:
### 서울 3박 4일 반나절 여행 스케줄 제안: 여행자를 위한 맞춤 가이드

#### **여행자 정보 및 기본 준비사항**
- **지역:** 서울
- **목적:** 역사와 현대 문화를 아우르는 체험
- **기간:** 4일 (반나절 집중 여행)
- **여행자 유형:** 타 지역 출신의 여행객
- **대안 동선 및 대중교통 이용 방법 포함**

---

### **Day 1: 경복궁 & 인사동 - 역사와 전통의 만남**

**오전**
- **경복궁 (입구 - 주요 건물 탐방)**
  - **장소:** 경복궁 (지하철 3호선 안국역 2번출구)
  - **이용시간:** 오전 9:30 - 정오
  - **특징:** 조선 시대의 상징적 건축물과 아름다운 정원 관람
  - **주의사항:** 코로나 방역 수칙 준수 (마스크 착용, 손 소독제 사용)
  - **대중교통 이용:** 지하철 안국역에서 도보 또는 도보 약 5분
  - **대안:** 비올 경우 실내 전시관인 경복궁 궁중박물관 방문

**오후**
- **인사동 전통 공예품 및 한복 쇼핑**
  - **장소:** 인사동 (지하철 3호선 종각역 인근)
  - **이용시간:** 오후 1시 - 오후 5시
  - **특징:** 한국 전통 공예품 구매 및 한복 대여
  - **주의사항:** 비올 경우 인근 카페나 실내 전시 공간에서 시간 보내기
  - **대중교통 이용:** 지하철 종각역에서 도보 약 10분

**저녁**
- **전통 음식점 방문**
  - **장소:** 경복궁 근처 (예: 고궁비빔밥, 불고기 전문점)
  - **이용시간:** 저녁 7시 - 9시
  - **특징:** 전통 한식 체험
  - **주의사항:** 비올 경우 인근 실내 맛집이나 전통차집 이용
  - **대중교통 이용:** 경복궁 내 가까운 지하철 역 이용 또는 도보

---

### **Day 2: 북악산 탐험 & 도심 쇼핑**

**오전**
- **북악산 등산**
  - **장소:** 북악산 정상 (북악산 등산 코스)
  - **이용시간:** 오전 9시 - 오후 1시
  - **특징:** 자연을 즐기며 서울의 멋진 전망 감상
  - **주의사항:** 비올 경우 등산 장비 대여점 인근 카페나 휴식 공간 방문
  - **대중교통 이용:** 북악산 입구 (지하철 2호선 동대문역 또는 3호선 동대문역 인근)에서 도보 약 20분

**오후**
- **등산 후 카페 및 지역 상권 탐방**
  - **장소:** 북악산 근처 카페 (예: 서울카페 투어 코스)
  - **이용시간:** 오후 2시 - 4시
  - **특징:** 휴식 및 지역 상권 체험
  - **주의사항:** 비올 경우 실내 관광지 또는 쇼핑몰 방문
  - **대중교통 이용:** 등산 코스 근처의 지하철 역 이용

**저녁**
- **신선한 해산물 레스토랑**
  - **장소:** 북부 지역 (예: 맛집 추천 사이트 활용)
  - **이용시간:** 저녁 7시 - 9시
  - **특징:** 현대 서울의 미식 문화 체험
  - **주의사항:** 비올 경우 실내 레스토랑이나 음식 배달 앱 이용
  - **대중교통 이용:** 북부 지역 지하철 역 이용

---

### **Day 3: 도심 관광과 현대적 쇼핑**

**오전**
- **명동/강남 쇼핑**
  - **장소:** 명동/강남 쇼핑몰 (예: 신세계백화점 강남점, 명동쇼핑몰)
  - **이용시간:** 오전 9시 - 정오
  - **특징:** 패션, electronics, 현대적 쇼핑 트렌드 체험
  - **주의사항:** 비올 경우 실내 쇼핑몰 이용 (예: 현대백화점)
  - **대중교통 이용:** 지하철 명동역 또는 강남역 이용

**오후**
- **역사적 장소 혹은 현대 쇼핑몰 방문**
  - **장소:** 서울역사박물관 또는 강남의 현대 쇼핑몰
  - **이용시간:** 오후 2시 - 오후 5시
  - **특징:** 역사와 현대의 접점 체험
  - **주의사항:** 비올 경우 실내 쇼핑몰 또는 갤러리 방문
  - **대중교통 이용:** 서울역사박물관 (지하철 3호선 동대문역 인근) 또는 강남지역 역 이용

**저녁**
- **서울 야경 감상**
  - **장소:** 홍대 근처 (예: 홍대 스카이라인, 야경 카페)
  - **이용시간:** 저녁 6시 - 8시
  - **특징:** 현대 서울의 야경과 문화 체험
  - **주의사항:** 비올 경우 실내 명소 방문 (예: 실내 미술관 또는 카페)
  - **대중교통 이용:** 지하철 홍대입구역 이용

---

### **추가 정보 및 주의사항**
- **대중교통:** 
  - **지하철:** 서울 지하철 시스템 (이용 설명서 확인 추천)
  - **택시/버스:** 주요 지역에서는 택시와 버스가 편리하며, 앱을 통해 예약 가능
  
- **비 오는 날 대비:**
  - **실내 장소:** 박물관, 카페, 실내 쇼핑몰 등
  - **예행:** 각 장소의 운영 시간 확인 및 미리 예약 (특히 식당 및 카페)

이러한 스케줄을 통해 서울의 깊이 있는 문화적 경험을 즐길 수 있으며, 지역 간 이동과 비 오는 날 대비 방안을 포함하여 유연하게 계획을 조정할 수 있습니다.**
```