{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/seirah-yang/BootCamp/blob/main/Graph%EC%8B%A4%EC%8A%B5.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DoRR-8vcpYIF"
      },
      "source": [
        "# 1. GraphRAG (with ollama)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "nX4tJKxTpaLF",
        "outputId": "15487747-a7ec-4bc6-8833-9a95f91c7d7b"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting colab-xterm\n",
            "  Downloading colab_xterm-0.2.0-py3-none-any.whl.metadata (1.2 kB)\n",
            "Requirement already satisfied: ptyprocess~=0.7.0 in /usr/local/lib/python3.12/dist-packages (from colab-xterm) (0.7.0)\n",
            "Requirement already satisfied: tornado>5.1 in /usr/local/lib/python3.12/dist-packages (from colab-xterm) (6.4.2)\n",
            "Downloading colab_xterm-0.2.0-py3-none-any.whl (115 kB)\n",
            "\u001b[?25l   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m0.0/115.6 kB\u001b[0m \u001b[31m?\u001b[0m eta \u001b[36m-:--:--\u001b[0m\r\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m115.6/115.6 kB\u001b[0m \u001b[31m5.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: colab-xterm\n",
            "Successfully installed colab-xterm-0.2.0\n"
          ]
        }
      ],
      "source": [
        "!pip install colab-xterm"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "iiN4pzlzialr"
      },
      "outputs": [],
      "source": [
        "%load_ext colabxterm"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RG8Kx43biafJ"
      },
      "outputs": [],
      "source": [
        "%xterm\n",
        "\n",
        "# 아래를 한줄씩 마우스 오른쪽 붙여넣기로\n",
        "# curl -fsSL https://ollama.com/install.sh | sh\n",
        "# ollama serve &\n",
        "# ollama pull exaone3.5:2.4b &\n",
        "# ollama pull nomic-embed-text &"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hoQ3LUrdU0av"
      },
      "outputs": [],
      "source": [
        "!pip install langchain_ollama"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jKL5d7TIU5d9"
      },
      "outputs": [],
      "source": [
        "!pip install langchain_community"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8cuO43viU-AO"
      },
      "outputs": [],
      "source": [
        "!pip install neo4j"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MiPW_WAAVluX"
      },
      "source": [
        "# GraphRAG: 1 hop 탐색"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "cjc5h62CibpI",
        "outputId": "54ac5149-5418-45e2-bad2-265963dc2504"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/tmp/ipython-input-2133391593.py:7: LangChainDeprecationWarning: The class `Neo4jGraph` was deprecated in LangChain 0.3.8 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-neo4j package and should be used instead. To use it run `pip install -U :class:`~langchain-neo4j` and import as `from :class:`~langchain_neo4j import Neo4jGraph``.\n",
            "  graph = Neo4jGraph(\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "질문: 당뇨병에 효과적인 식단은 무엇인가요?\n",
            "\n",
            "\n",
            "\u001b[1m> Entering new GraphCypherQAChain chain...\u001b[0m\n",
            "Generated Cypher:\n",
            "\u001b[32;1m\u001b[1;3mcypher\n",
            "MATCH (d:Disease {name: 'Diabetes'})-[:TREATED_BY]->(di:Diet)\n",
            "RETURN di.description AS EffectiveDietDescription\n",
            "\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N50', status_description='warn: label does not exist. The label `Diet` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=57, offset=63>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 63, 'line': 2, 'column': 57}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease {name: 'Diabetes'})-[:TREATED_BY]->(di:Diet)\\nRETURN di.description AS EffectiveDietDescription\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N50', status_description='warn: label does not exist. The label `Disease` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=10, offset=16>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 16, 'line': 2, 'column': 10}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease {name: 'Diabetes'})-[:TREATED_BY]->(di:Diet)\\nRETURN di.description AS EffectiveDietDescription\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N52', status_description='warn: property key does not exist. The property `name` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=19, offset=25>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 25, 'line': 2, 'column': 19}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease {name: 'Diabetes'})-[:TREATED_BY]->(di:Diet)\\nRETURN di.description AS EffectiveDietDescription\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N51', status_description='warn: relationship type does not exist. The relationship type `TREATED_BY` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=40, offset=46>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 46, 'line': 2, 'column': 40}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease {name: 'Diabetes'})-[:TREATED_BY]->(di:Diet)\\nRETURN di.description AS EffectiveDietDescription\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N52', status_description='warn: property key does not exist. The property `description` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=3, column=11, offset=79>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 79, 'line': 3, 'column': 11}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease {name: 'Diabetes'})-[:TREATED_BY]->(di:Diet)\\nRETURN di.description AS EffectiveDietDescription\\n\"\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Full Context:\n",
            "\u001b[32;1m\u001b[1;3m[]\u001b[0m\n",
            "\n",
            "\u001b[1m> Finished chain.\u001b[0m\n",
            "답변: 당뇨병 관리에 효과적인 식단은 균형 잡힌 선택을 기반으로 합니다. 주로 다음과 같은 요소들이 포함됩니다:\n",
            "\n",
            "1. **다양한 채소와 과일**: 비타민과 식이섬유가 풍부하여 혈당 조절에 도움을 줍니다.\n",
            "2. **전곡류**: 현미, 통밀빵 등은 복합 탄수화물로 천천히 소화되어 혈당 상승을 완만하게 합니다.\n",
            "3. **건강한 지방**: 아보카도, 견과류, 올리브 오일 같은 불포화 지방산은 포만감을 주고 다른 영양소 공급에 좋습니다.\n",
            "4. **단백질**: 저지방 고기, 어류, 콩류 등은 단백질 공급원으로 포만감을 주며 혈당 관리에 도움이 됩니다.\n",
            "5. **수분 섭취**: 충분한 물 섭취는 체내 독소 제거와 혈당 수준 안정에 중요합니다.\n",
            "\n",
            "전문가와 상의하여 개인의 건강 상태와 필요에 맞는 맞춤형 식단을 계획하는 것이 가장 좋습니다.\n",
            "\n",
            "질문: 고혈압을 치료하는 식단 중 효과 점수가 8 이상인 것은?\n",
            "\n",
            "\n",
            "\u001b[1m> Entering new GraphCypherQAChain chain...\u001b[0m\n",
            "Generated Cypher:\n",
            "\u001b[32;1m\u001b[1;3mcypher\n",
            "MATCH (d:Diet)-[t:TREATED_BY]->(di:Disease {name: '고혈압'})\n",
            "WHERE t.effectiveness >= 8\n",
            "RETURN d.description AS DietDescription\n",
            "\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N52', status_description='warn: property key does not exist. The property `effectiveness` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=3, column=9, offset=73>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 73, 'line': 3, 'column': 9}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Diet)-[t:TREATED_BY]->(di:Disease {name: '고혈압'})\\nWHERE t.effectiveness >= 8\\nRETURN d.description AS DietDescription\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N52', status_description='warn: property key does not exist. The property `description` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=4, column=10, offset=101>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 101, 'line': 4, 'column': 10}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Diet)-[t:TREATED_BY]->(di:Disease {name: '고혈압'})\\nWHERE t.effectiveness >= 8\\nRETURN d.description AS DietDescription\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N50', status_description='warn: label does not exist. The label `Disease` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=36, offset=42>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 42, 'line': 2, 'column': 36}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Diet)-[t:TREATED_BY]->(di:Disease {name: '고혈압'})\\nWHERE t.effectiveness >= 8\\nRETURN d.description AS DietDescription\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N50', status_description='warn: label does not exist. The label `Diet` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=10, offset=16>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 16, 'line': 2, 'column': 10}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Diet)-[t:TREATED_BY]->(di:Disease {name: '고혈압'})\\nWHERE t.effectiveness >= 8\\nRETURN d.description AS DietDescription\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N51', status_description='warn: relationship type does not exist. The relationship type `TREATED_BY` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=19, offset=25>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 25, 'line': 2, 'column': 19}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Diet)-[t:TREATED_BY]->(di:Disease {name: '고혈압'})\\nWHERE t.effectiveness >= 8\\nRETURN d.description AS DietDescription\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N52', status_description='warn: property key does not exist. The property `name` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=45, offset=51>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 51, 'line': 2, 'column': 45}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Diet)-[t:TREATED_BY]->(di:Disease {name: '고혈압'})\\nWHERE t.effectiveness >= 8\\nRETURN d.description AS DietDescription\\n\"\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Full Context:\n",
            "\u001b[32;1m\u001b[1;3m[]\u001b[0m\n",
            "\n",
            "\u001b[1m> Finished chain.\u001b[0m\n",
            "답변: I don't have specific information provided about dietary treatments for hypertension or their effectiveness scores to accurately identify which diets score 8 or higher in effectiveness. For detailed advice on hypertension management through diet, consulting healthcare professionals specializing in nutrition or cardiology would be advisable.\n",
            "\n",
            "질문: 지중해식 식단이 어떤 질병에 효과적인가요?\n",
            "\n",
            "\n",
            "\u001b[1m> Entering new GraphCypherQAChain chain...\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N50', status_description='warn: label does not exist. The label `Disease` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=10, offset=16>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 16, 'line': 2, 'column': 10}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease)-[r:TREATED_BY]->(di:Diet {description: '지중해식 식단'})\\nRETURN d.name AS DiseaseName\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N50', status_description='warn: label does not exist. The label `Diet` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=39, offset=45>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 45, 'line': 2, 'column': 39}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease)-[r:TREATED_BY]->(di:Diet {description: '지중해식 식단'})\\nRETURN d.name AS DiseaseName\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N52', status_description='warn: property key does not exist. The property `description` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=45, offset=51>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 51, 'line': 2, 'column': 45}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease)-[r:TREATED_BY]->(di:Diet {description: '지중해식 식단'})\\nRETURN d.name AS DiseaseName\\n\"\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Generated Cypher:\n",
            "\u001b[32;1m\u001b[1;3mcypher\n",
            "MATCH (d:Disease)-[r:TREATED_BY]->(di:Diet {description: '지중해식 식단'})\n",
            "RETURN d.name AS DiseaseName\n",
            "\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N51', status_description='warn: relationship type does not exist. The relationship type `TREATED_BY` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=2, column=22, offset=28>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 28, 'line': 2, 'column': 22}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease)-[r:TREATED_BY]->(di:Diet {description: '지중해식 식단'})\\nRETURN d.name AS DiseaseName\\n\"\n",
            "WARNING:neo4j.notifications:Received notification from DBMS server: <GqlStatusObject gql_status='01N52', status_description='warn: property key does not exist. The property `name` does not exist in database `neo4j`. Verify that the spelling is correct.', position=<SummaryInputPosition line=3, column=10, offset=85>, raw_classification='UNRECOGNIZED', classification=<NotificationClassification.UNRECOGNIZED: 'UNRECOGNIZED'>, raw_severity='WARNING', severity=<NotificationSeverity.WARNING: 'WARNING'>, diagnostic_record={'_classification': 'UNRECOGNIZED', '_severity': 'WARNING', '_position': {'offset': 85, 'line': 3, 'column': 10}, 'OPERATION': '', 'OPERATION_CODE': '0', 'CURRENT_SCHEMA': '/'}> for query: \"cypher\\nMATCH (d:Disease)-[r:TREATED_BY]->(di:Diet {description: '지중해식 식단'})\\nRETURN d.name AS DiseaseName\\n\"\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Full Context:\n",
            "\u001b[32;1m\u001b[1;3m[]\u001b[0m\n",
            "\n",
            "\u001b[1m> Finished chain.\u001b[0m\n",
            "답변: 지중해식 식단은 심혈관 질환, 일부 유형의 암(예: 대장암), 당뇨병, 그리고 고혈압과 같은 만성 질환 관리에 긍정적인 영향을 미치는 것으로 알려져 있습니다. 이 식단은 과일, 채소, 전곡류, 건강한 지방(예: 올리브 오일), 그리고 적은 양의 단백질과 유제품을 강조함으로써 전반적인 건강 증진에 도움을 줍니다. 하지만 개인의 건강 상태에 따라 효과는 다를 수 있으므로 전문가와 상의하는 것이 좋습니다.\n"
          ]
        }
      ],
      "source": [
        "from langchain_ollama import ChatOllama\n",
        "from langchain_community.graphs import Neo4jGraph\n",
        "from langchain.chains import GraphCypherQAChain\n",
        "from langchain.prompts import PromptTemplate\n",
        "\n",
        "# 1. Neo4j Sandbox 연결 설정\n",
        "graph = Neo4jGraph(\n",
        "    url=\"bolt://44.199.211.150:7687\",  # 샌드박스 Bolt URL로 교체\n",
        "    username=\"neo4j\",\n",
        "    password=\"compromise-curtains-watts\"      # 샌드박스 비밀번호로 교체\n",
        ")\n",
        "\n",
        "# 2. Ollama의 gemma2 모델 설정\n",
        "llm = ChatOllama(\n",
        "    model=\"exaone3.5:2.4b\",\n",
        "    base_url=\"http://localhost:11434\",    # Ollama 기본 URL\n",
        "    temperature=0.7                       # 창의성 조절 (0~1)\n",
        ")\n",
        "\n",
        "# 3. Cypher 쿼리 생성 및 답변을 위한 프롬프트 템플릿\n",
        "CYPHER_GENERATION_TEMPLATE = \"\"\"\n",
        "당신은 Neo4j 그래프 데이터베이스에서 Cypher 쿼리를 생성하는 전문가입니다.\n",
        "사용자의 질문에 따라 적절한 Cypher 쿼리를 작성하세요.\n",
        "질문: {question}\n",
        "\n",
        "그래프 스키마:\n",
        "- 노드: Disease(name), Diet(description)\n",
        "- 관계: TREATED_BY(effectiveness)\n",
        "\n",
        "Cypher 쿼리만 반환하세요.\n",
        "\"\"\"\n",
        "\n",
        "cypher_prompt = PromptTemplate(\n",
        "    input_variables=[\"question\"],\n",
        "    template=CYPHER_GENERATION_TEMPLATE\n",
        ")\n",
        "\n",
        "# 4. GraphRAG 체인 생성\n",
        "chain = GraphCypherQAChain.from_llm(\n",
        "    llm=llm,\n",
        "    graph=graph,\n",
        "    cypher_prompt=cypher_prompt,\n",
        "    verbose=True,  # 쿼리 실행 과정 확인용\n",
        "    allow_dangerous_requests=True\n",
        ")\n",
        "\n",
        "# 5. 질문 실행 함수\n",
        "def ask_question(query):\n",
        "    response = chain.invoke({\"query\": query}) #LLM에게 질문을 던짐 -> LLM은 프롬프트를 입력 받고 cypher query를 만듦\n",
        "                                                                                # -> query를 neo4j에게 던지면 그래프에서 당뇨병에 대한 연결정보를 가져옴\n",
        "                                                                                # 가져온 정보를 참고해서 대답\n",
        "    return response[\"result\"]\n",
        "\n",
        "# 6. 예시 질문 실행\n",
        "questions = [\n",
        "    \"당뇨병에 효과적인 식단은 무엇인가요?\",\n",
        "    \"고혈압을 치료하는 식단 중 효과 점수가 8 이상인 것은?\",\n",
        "    \"지중해식 식단이 어떤 질병에 효과적인가요?\"\n",
        "]\n",
        "\n",
        "for q in questions:\n",
        "    print(f\"\\n질문: {q}\")\n",
        "    answer = ask_question(q)\n",
        "    print(f\"답변: {answer}\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "YG5SPz2WVqjH"
      },
      "source": [
        "# # GraphRAG: 2 hop 탐색"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vxBuiAQZUzPY"
      },
      "outputs": [],
      "source": [
        "# 3. Cypher 쿼리 생성 프롬프트 템플릿 (2홉 강화)\n",
        "CYPHER_GENERATION_TEMPLATE = \"\"\"\n",
        "당신은 Neo4j 그래프 데이터베이스에서 Cypher 쿼리를 생성하는 전문가입니다.\n",
        "사용자의 질문에 따라 적절한 Cypher 쿼리를 작성하세요. 2홉 경로도 고려하세요.\n",
        "모든 변수를 `WITH` 절에서 명시적으로 전달하여 변수가 누락되지 않도록 하세요.\n",
        "질문: {question}\n",
        "\n",
        "그래프 스키마:\n",
        "- 노드: Disease(name), Diet(description), Nutrient(name)\n",
        "- 관계: TREATED_BY(effectiveness), CONTAINS(amount)\n",
        "\n",
        "Cypher 쿼리만 반환하세요。\n",
        "\"\"\"\n",
        "cypher_prompt = PromptTemplate(\n",
        "    input_variables=[\"question\"],\n",
        "    template=CYPHER_GENERATION_TEMPLATE\n",
        ")\n",
        "\n",
        "# 4. GraphRAG 체인 생성\n",
        "chain = GraphCypherQAChain.from_llm(\n",
        "    llm=llm,\n",
        "    graph=graph,\n",
        "    cypher_prompt=cypher_prompt,\n",
        "    verbose=True,\n",
        "    allow_dangerous_requests=True\n",
        ")\n",
        "\n",
        "# 5. 질문 실행\n",
        "query = \"Diabetes에 효과적인 식단은 무엇인가요?\"\n",
        "response = chain.invoke({\"query\": query})\n",
        "print(f\"질문: {query}\")\n",
        "print(f\"답변: {response['result']}\")"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}