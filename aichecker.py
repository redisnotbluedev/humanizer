import aiohttp, asyncio

class AIChecker:
	def __init__(self):
		self.session = None
		self.providers = [self.zerogpt]
	
	async def _ensure_session(self):
		if self.session is None:
			self.session = aiohttp.ClientSession(headers={
				"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:146.0) Gecko/20100101 Firefox/146.0"
			})
	
	async def check_async(self, text):
		results = await asyncio.gather(*[p(text) for p in self.providers])
		scores = []
		sentence_scores = {}
		providers = []
		
		for r in results:
			if r["status"] == "failed":
				continue

			providers.append(r["provider"])
			scores.append(r["score"])
			for s in r["sentences"]:
				sentence_text = s["text"]
				sentence_score = s["score"]
				
				if sentence_text not in sentence_scores:
					sentence_scores[sentence_text] = []
				sentence_scores[sentence_text].append(sentence_score)
		
		fake_sentences = []
		for sentence_text, score_list in sentence_scores.items():
			avg_score = sum(score_list) / len(score_list)
			if avg_score > 50:
				fake_sentences.append(sentence_text)

		return {
			"score": sum(scores) / len(scores),
			"fake_sentences": fake_sentences,
			"text": text,
			"providers": providers
		}
	
	async def zerogpt(self, text):
		await self._ensure_session()

		max_retries = 5
		base_delay = 1
		
		for attempt in range(max_retries):
			try:
				response = await self.session.post(
					"https://api.zerogpt.com/api/detect/detectText",
					json={"input_text": text},
					headers={
						"Referer": "https://www.zerogpt.com/",
						"Origin": "https://www.zerogpt.com"
					},
					timeout=aiohttp.ClientTimeout(total=30),
					ssl=False
				)
				
				if response.status == 429:
					if attempt < max_retries - 1:
						delay = base_delay * (2 ** attempt)
						await asyncio.sleep(delay)
						continue
					else:
						return {"status": "failed"}
				
				response.raise_for_status()
				data = await response.json()
				
				if not data or "data" not in data:
					if attempt < max_retries - 1:
						delay = base_delay * (2 ** attempt)
						await asyncio.sleep(delay)
						continue
					else:
						return {"status": "failed"}
	
				ai_sentences = data["data"]["h"]
				human_sentences = data["data"].get("sentences", [])

				sentences = []
				sentences.extend([{"text": s.strip(), "score": 100} for s in ai_sentences])
				sentences.extend([{"text": s.strip(), "score": 0} for s in human_sentences])

				return {
					"status": "success",
					"provider": "zerogpt",
					"score": data["data"]["fakePercentage"],
					"sentences": sentences,
					"text": text
				}

			except (aiohttp.ClientError, asyncio.TimeoutError, KeyError):
				if attempt < max_retries - 1:
					delay = base_delay * (2 ** attempt)
					await asyncio.sleep(delay)
					continue
				else:
					return {"status": "failed"}
		
		return {"status": "failed"}

	async def originality(self, text):
		await self._ensure_session()

		max_retries = 5
		base_delay = 1
		
		for attempt in range(max_retries):
			try:
				resp = await self.session.post(
					"https://api.originality.ai/api/v2-tools/free-tools/ai-scan",
					json={"content": text},
					headers={
						"Referer": "https://corefreetools.originality.ai",
						"Origin": "https://corefreetools.originality.ai"
					},
					timeout=aiohttp.ClientTimeout(total=30)
				)

				resp.raise_for_status()
				data = await resp.json()
				
				if not data:
					if attempt < max_retries - 1:
						delay = base_delay * (2 ** attempt)
						await asyncio.sleep(delay)
						continue
					else:
						return {"status": "failed"}
				
				blocks = data.get("blocks", [])
				
				total_ai_score = 0
				sentence_count = 0
				sentences = []
				
				for block in blocks:
					block_text = block["text"].strip()
					if not block_text:
						continue
					
					result = block.get("result", {})
					fake_score = result.get("fake", 0)
					
					ai_percentage = fake_score * 100
					
					sentences.append({
						"text": block_text,
						"score": ai_percentage
					})
					
					total_ai_score += ai_percentage
					sentence_count += 1
				
				overall_score = total_ai_score / sentence_count if sentence_count > 0 else 0
				
				return {
					"status": "success",
					"provider": "originality",
					"score": overall_score,
					"sentences": sentences,
					"text": text
				}
			except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as e:
				if attempt < max_retries - 1:
					delay = base_delay * (2 ** attempt)
					await asyncio.sleep(delay)
					continue
				else:
					return {"status": "failed"}
		
		return {"status": "failed"}

	async def close(self):
		if self.session:
			await self.session.close()

if __name__ == "__main__":
	async def a():
		c = AIChecker()
		t = open("essay.txt").read()
		j = []
		for _ in range(10):
			j.append(c.check_async(t))
		b = await asyncio.gather(*j)
		for r in b:
			print(f"Score: {r["score"]}, false sentences: {len(r["fake_sentences"])}, providers: {r["providers"]}")
		await c.close()

	asyncio.run(a())