
📌 Accuracy 의미 요약

해당 코드에서 accuracy는:

\text{Accuracy} = \frac{\text{문서 내부 evidence로 지지(entail)된 claim 수}}{\text{전체 claim 수}}

섹션의 주장들이 얼마나 자기 근거로 뒷받침되는가를 보는 지표
	•	높다 = 대부분의 claim이 문서 내부에서 근거와 일치한다
	•	낮다 = claim이 모순되거나 근거가 없어 문서 신뢰성이 떨어진다

1. entail (지지된 주장 수)
	•	NLIModel.predict 함수에서 claim 문장과 evidence 문장을 비교했을 때
→ 숫자 단위 일치, 키워드 유사성 등을 근거로
→ 해당 claim이 evidence에 의해 지지(entailment) 된다고 판정된 경우 카운트.

“이 문장에서 주장한 KPI/사실이 문서 내부 다른 문장으로 뒷받침된다”는 의미.

⸻

2. contra (모순된 주장 수)
	•	같은 비교에서 claim과 evidence가 숫자 충돌하거나 논리적으로 맞지 않음으로 판정된 경우 카운트.
	•	예: claim에서 “정확도 95%”라 했는데 evidence에서는 “정확도 70%”라고 하면 contra++

즉, “문서 내에서 자기모순/충돌하는 주장”을 의미.

⸻

3. unknown (중립/판정불가 주장 수)
	•	evidence와 유사성은 있지만 숫자 불일치나 명확한 entail/contradiction 신호가 없는 경우
	•	또는 confidence가 낮아 판단을 보류한 경우 카운트.

“문서 내부에서 확인되지도 않고 부정되지도 않는 애매한 주장”을 의미.

⸻

4. tot (전체 claim 수)

\text{tot} = \text{entail} + \text{contra} + \text{unknown}
	•	섹션 안에서 뽑아낸 claim(=KPI 들어간 문장들)의 총 개수
