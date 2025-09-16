!pip install langchain_openai  # crew AI 를 사용하기 위해 langchain_openai가 필요
!pip install crewai
!pip install colab-xterm

%load_ext colabxterm

## !curl -fsSL https://ollama.com/install.sh | sh # COLAB에서 할 땐 아래 가상환경에서 쓴다. 
%xterm

# 아래를 한줄씩 마우스 오른쪽 붙여넣기로
# curl -fsSL https://ollama.com/install.sh | sh
# ollama serve & ollama pull exaone3.5:2.4b

from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# 크류 AI는 Open API 키가 있어야 한다. 그래서 NA라는 Fake를 줄 것이다.
os.environ["OPENAI_API_KEY"] = "NA"

# 에이전트로 사용할 LLM 정의
llm = ChatOpenAI(
    model="ollama/exaone3.5:2.4b", #ollama pull exaone3.5:2.4b 사용
    base_url="http://localhost:11434"  # 내 컴퓨터 11434 포트 (올라마 서버)에 있음
)

# 콘텐츠 기획자 에이전트=역할 정의(3개를 만든다.)
planner = Agent(
    role="콘텐츠 기획자",
    goal="주제 {topic}에 대해 흥미롭고 사실에 기반한 콘텐츠를 기획한다. 모든 결과는 한국어로 작성합니다.",
    backstory="당신은 'https://medium.com/'에서 주제 {topic}에 관한 블로그 글 기획 작업을 진행 중입니다. "
              "독자들이 유익한 정보를 얻고 올바른 결정을 내릴 수 있도록 도와주는 정보를 수집합니다. "
              "자세한 개요와 관련 주제, 하위 주제들을 준비하세요. 이 작업은 콘텐츠 작가가 글을 작성하는 기반이 됩니다. "
              "답변은 반드시 한국어로 작성됩니다.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 콘텐츠 작가 에이전트 정의 = 의견기고문
writer = Agent(
    role="콘텐츠 작가",
    goal="주제 {topic}에 대한 통찰력 있고 사실에 기반한 의견 기고문을 작성한다. 모든 결과는 한국어로 작성합니다.",
    backstory="당신은 'https://medium.com/'에서 주제 {topic}에 관한 새로운 의견 기고문을 작성 중입니다. "
              "콘텐츠 기획자가 제공한 개요와 관련 정보를 기반으로 글을 작성하세요. "
              "개요의 주요 목표와 방향을 따르며, 객관적이고 공정한 통찰을 제공하고, 출처를 명확히 하세요. "
              "의견과 객관적 사실을 구분해야 합니다. 답변은 반드시 한국어로 작성됩니다.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 편집자 에이전트 정의 = 작성된 글을 편집/작성
editor = Agent(
    role="편집자",
    goal="주어진 블로그 글을 'https://medium.com/'의 글쓰기 스타일에 맞게 편집한다. 모든 결과는 한국어로 작성합니다.",
    backstory="당신은 콘텐츠 작가로부터 전달받은 블로그 글을 검토하는 편집자입니다. "
              "블로그 글이 언론의 최선의 관행을 따르고, 의견이나 주장이 균형 잡힌 시각을 제공하며, "
              "논란의 여지가 있는 주제나 의견은 피하도록 교정하세요. 결과는 반드시 꼭 한국어로 작성됩니다.",
    llm=llm,
    allow_delegation=False,
    verbose=True
)

# 콘텐츠 기획 태스크 정의 = 작업수행 해야하는 작업들에 대해 기술한다.
plan = Task(
    description=(
        "1. 주제 {topic}와 관련된 최신 트렌드, 주요 인물, 그리고 주목할 만한 뉴스를 우선 파악합니다.\n"
        "2. 대상 독자를 분석하여, 그들의 관심사와 문제점을 고려합니다.\n"
        "3. 서론, 핵심 포인트, 행동 촉구(Call to Action)를 포함한 상세 콘텐츠 개요를 작성합니다.\n"
        "4. SEO 키워드와 관련 데이터, 출처를 포함합니다."
    ),
    expected_output="개요, 독자 분석, SEO 키워드 및 참고 자료가 포함된 포괄적인 콘텐츠 기획 문서",
    agent=planner,
)

# 콘텐츠 작성 태스크 정의
write = Task(
    description=(
        "1. 콘텐츠 기획 문서를 바탕으로 주제 {topic}에 관한 흥미로운 블로그 글을 작성합니다.\n"
        "2. SEO 키워드를 자연스럽게 포함합니다.\n"
        "3. 각 섹션 및 부제목을 매력적으로 구성합니다.\n"
        "4. 흥미로운 서론, 통찰력 있는 본문, 그리고 요약 결론으로 구성된 글의 구조를 갖춥니다.\n"
        "5. 문법 오류 및 브랜드의 목소리에 맞게 교정합니다.\n"
    ),
    expected_output="출판 준비가 된, 마크다운 형식의 잘 작성된 블로그 글 (각 섹션은 2~3 단락 포함)",
    agent=writer,
)

# 블로그 글 편집 태스크 정의
edit = Task(
    description="주어진 블로그 글을 문법 오류 및 브랜드의 목소리에 맞게 교정합니다.",
    expected_output="출판 준비가 된, 마크다운 형식의 잘 작성된 블로그 글 (각 섹션은 2~3 단락 포함)",
    agent=editor
)

# 에이전트와 태스크를 포함하는 크루 생성
crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit],
    verbose=True
)

# 입력값 설정 및 크루 실행
inputs = {"topic": "LangGraph, Autogen, Crewai를 활용한 멀티 에이전트 시스템 구축 비교 연구"}
result = crew.kickoff(inputs=inputs)


final_answer= result.dict()['raw']
print(final_answer)
