import os, random, asyncio, dotenv, httpx
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError
from aichecker import AIChecker

dotenv.load_dotenv()

ESSAY_FILE = "essay.txt"
PROMPT_FILE = "prompt.txt"
CHOICES = 7
MAX_ROLLBACKS = 3
TARGET_SCORE = 30

checker = AIChecker()
generator = AsyncOpenAI(
	base_url="https://api.mapleai.de/v1",
	api_key=os.getenv("API_KEY"),
	max_retries=5,
	timeout=60.0,
	http_client=httpx.AsyncClient(verify=False)
)

async def api_call_with_backoff(coro_func, max_retries=5):
	"""Execute an API call with exponential backoff for rate limits"""
	for attempt in range(max_retries):
		try:
			return await coro_func()
		except RateLimitError as e:
			if attempt == max_retries - 1:
				raise
			wait_time = (2 ** attempt) + random.uniform(0, 1)
			print(f"  Rate limited, waiting {wait_time:.1f}s...")
			await asyncio.sleep(wait_time)
		except APIConnectionError as e:
			if attempt == max_retries - 1:
				raise
			wait_time = (2 ** attempt)
			print(f"  Connection error, retrying in {wait_time}s...")
			await asyncio.sleep(wait_time)
		except APIError as e:
			print(f"  API error: {e}")
			raise

async def batch_requests(tasks, batch_size=50, delay=2):
	results = []
	for i in range(0, len(tasks), batch_size):
		batch = tasks[i:i + batch_size]
		try:
			batch_results = await asyncio.gather(*batch, return_exceptions=True)
			
			# Check for exceptions
			for j, result in enumerate(batch_results):
				if isinstance(result, Exception):
					print(f"  Error in request {i+j}: {result}")
					raise result
			
			results.extend(batch_results)
		except Exception as e:
			print(f"  Batch failed: {e}")
			raise
		
		if i + batch_size < len(tasks):
			print(f"  Processed {i + batch_size}/{len(tasks)} requests, pausing...")
			await asyncio.sleep(delay)
	return results

def smart_batch_sentences(text, fake_sentences, max_batch_size=4):
	"""Batch sentences that are adjacent in the original text"""
	# Find positions of flagged sentences
	sentence_positions = []
	for sentence in fake_sentences:
		pos = text.find(sentence)
		sentence_positions.append((pos, sentence))
	
	# Sort by position
	sentence_positions.sort()
	
	# Group adjacent sentences
	batches = []
	current_batch = [sentence_positions[0][1]]
	
	for i in range(1, len(sentence_positions)):
		prev_pos, prev_sent = sentence_positions[i-1]
		curr_pos, curr_sent = sentence_positions[i]
		
		# If sentences are close together (within 200 chars) and batch isn't full
		if curr_pos - (prev_pos + len(prev_sent)) < 200 and len(current_batch) < max_batch_size:
			current_batch.append(curr_sent)
		else:
			batches.append(" ".join(current_batch))
			current_batch = [curr_sent]
	
	batches.append(" ".join(current_batch))
	return batches

def extract_paragraphs_with_flagged(text, fake_sentences):
	"""Extract paragraphs that contain flagged sentences"""
	paragraphs = text.split('\n\n')
	flagged_paragraphs = []
	
	for para in paragraphs:
		if any(sentence in para for sentence in fake_sentences):
			flagged_paragraphs.append(para)
	
	return flagged_paragraphs

async def main():
	try:
		max_tokens = 0
		with open(ESSAY_FILE, "r", encoding="utf-8") as f:
			current_text = f.read()
			# 20 extra for breathing space
			max_tokens = int(len(current_text.split()) * 1.3 + 20)
		with open(PROMPT_FILE, "r", encoding="utf-8") as f:
			prompt = f.read()
		
		# Initial check
		initial_result = await checker.check_async(current_text)
		previous_score = initial_result["score"]
		current_fake_sentences = initial_result["fake_sentences"]
		
		# Track rewrite history per sentence
		sentence_history = {}  # original_sentence -> list of (round_num, score, rewrite)
		
		# Track best state globally
		best_ever_text = current_text
		best_ever_score = previous_score
		best_ever_round = 0
		consecutive_rollbacks = 0
		failure_history = ""
		
		# Track score history
		score_history = []
		
		print(f"Initial score: {previous_score:.2f}% ({len(current_fake_sentences)} flagged sentences)")
		
		round_num = 0
		while True:
			print(f"\n{'='*50}")
			print(f"Round {round_num + 1}")
			print(f"{'='*50}")
			
			if not current_fake_sentences:
				print("✓ No flagged sentences remaining!")
				break
			
			if previous_score <= TARGET_SCORE:
				print(f"✓ Target score reached!")
				break
			
			if consecutive_rollbacks >= MAX_ROLLBACKS:
				print(f"\n✗ Maximum consecutive rollbacks ({MAX_ROLLBACKS}) reached. Stopping.")
				break
			
			# Adjust strategy based on score
			if previous_score > 80:
				temp_min, temp_max = 1.2, 1.5
				strategy = "Try completely different phrasing, sentence structure, and vocabulary."
			elif previous_score > 50:
				temp_min, temp_max = 1.0, 1.3
				strategy = "Make substantial changes while keeping the core meaning."
			else:
				temp_min, temp_max = 0.9, 1.1
				strategy = "Make subtle stylistic adjustments to reduce AI patterns."
			
			# Increase temperature more aggressively after failure
			temp_boost = 0.2 * consecutive_rollbacks
			temp_min += temp_boost
			temp_max += temp_boost
			
			print(f"Targeting {len(current_fake_sentences)} flagged sentences...")
			print(f"Strategy: {strategy}")
			
			# Generate candidates sequentially
			candidates = []
			for candidate_num in range(CHOICES):
				print(f"  Generating candidate {candidate_num + 1}/{CHOICES}...")
				
				# Generate all sentence rewrites for THIS candidate
				rewrite_tasks = []
				rewrite_map = {}  # Track original -> rewrite mapping
				
				for sentence in current_fake_sentences:
					temp = random.uniform(temp_min, temp_max)
					
					# Build score history context
					score_context = ""
					if score_history:
						score_context = "\n\nOVERALL SCORE HISTORY:\n"
						for entry in score_history[-5:]:  # Last 5 rounds
							status = "✓ ACCEPTED" if entry["accepted"] else "↩ ROLLED BACK"
							score_context += f"  Round {entry['round']}: {entry['best_candidate_score']:.1f}% - {status}\n"
						score_context += f"Current best: {best_ever_score:.1f}% (Round {best_ever_round})"
					
					# Build history context for this specific sentence
					history_context = ""
					if sentence in sentence_history:
						history_context = "\n\nPREVIOUS REWRITES OF THIS SENTENCE (all flagged as AI):\n"
						for attempt_num, score, prev_rewrite in sentence_history[sentence][-3:]:  # Last 3 attempts
							history_context += f"  Attempt {attempt_num} ({score:.1f}% AI): {prev_rewrite}\n"
						history_context += "\nThese patterns didn't work. Try a completely different approach."
					
					rewrite_tasks.append(
						api_call_with_backoff(
							lambda s=sentence, t=temp, ct=current_text, p=prompt, ps=previous_score, hc=history_context, sc=score_context, fh=failure_history, strat=strategy: generator.chat.completions.create(
								messages=[
									{
										"role": "system",
										"content": f"""{p}

CURRENT TASK:
Rewrite this flagged sentence. Detection score: {ps:.1f}%

DETECTED ISSUES:
- AI-typical patterns in phrasing
- Predictable word choices for this formality level
- Mechanically perfect grammar

YOUR GOAL:
Rewrite to match a human writing, not an AI mimicking its tone.
Keep the formality EXACTLY the same, but make the execution less robotic.

{sc}

{hc if hc else ""}

{fh if fh else ""}

STRATEGY: {strat}

Output ONLY the rewritten sentence."""
									},
									{
										"role": "user",
										"content": f"Full text:\n{ct}\n\nRewrite:\n{s}"
									}
								],
								temperature=t,
								top_p=0.95,
								model="gemini-2.5-flash",
								max_tokens=max_tokens
							)
						)
					)
				
				# Process with batching
				rewrites = await batch_requests(rewrite_tasks, batch_size=40, delay=2)
				
				# Build candidate text and track rewrites
				candidate_text = current_text
				for i, sentence in enumerate(current_fake_sentences):
					new_sentence = rewrites[i].choices[0].message.content.strip()
					rewrite_map[sentence] = new_sentence
					candidate_text = candidate_text.replace(sentence, new_sentence, 1)
				
				candidates.append({
					"text": candidate_text,
					"rewrites": rewrite_map
				})
				
				# Breathing room between candidates
				if candidate_num < CHOICES - 1:
					await asyncio.sleep(2)
			
			# Check all candidates at once
			print("\nChecking all candidates...")
			check_tasks = [checker.check_async(c["text"]) for c in candidates]
			results = await asyncio.gather(*check_tasks)
			
			# Find best candidate THIS round
			best_idx = min(range(len(results)), key=lambda i: results[i]["score"])
			best_result = results[best_idx]
			best_candidate = candidates[best_idx]
			
			for i, result in enumerate(results):
				marker = " ← best" if i == best_idx else ""
				print(f"  Candidate {i+1}: {result['score']:.2f}% ({len(result['fake_sentences'])} flagged){marker}")
			
			# Update history with what got flagged
			for i, candidate in enumerate(candidates):
				result = results[i]
				for orig_sentence, rewrite in candidate["rewrites"].items():
					# If this rewrite is still flagged, record it
					if rewrite in result["fake_sentences"]:
						if orig_sentence not in sentence_history:
							sentence_history[orig_sentence] = []
						sentence_history[orig_sentence].append((round_num + 1, result["score"], rewrite))
			
			# Check if this is actually an improvement over our BEST EVER
			improvement = best_ever_score - best_result["score"]
			
			# Track score history
			score_history.append({
				"round": round_num + 1,
				"best_candidate_score": best_result["score"],
				"accepted": best_result["score"] < best_ever_score,
				"rolled_back": best_result["score"] >= best_ever_score
			})
			
			if best_result["score"] < best_ever_score:
				# Any improvement is good
				best_ever_text = best_candidate["text"]
				best_ever_score = best_result["score"]
				best_ever_round = round_num + 1
				consecutive_rollbacks = 0
				failure_history = ""
				
				print(f"\n✓ New best: {best_ever_score:.1f}% (round {best_ever_round})")
			else:
				# Actual regression
				consecutive_rollbacks += 1
				
				print(f"\n⚠ Round degraded quality: {previous_score:.1f}% → {best_result['score']:.1f}%")
				print(f"  Rolling back to round {best_ever_round} state ({best_ever_score:.1f}%)")
				print(f"  Consecutive rollbacks: {consecutive_rollbacks}/{MAX_ROLLBACKS}")
				
				# ROLLBACK
				current_text = best_ever_text
				previous_score = best_ever_score
				rollback_result = await checker.check_async(current_text)
				current_fake_sentences = rollback_result["fake_sentences"]
				
				# Add failure context for next attempt
				failure_history = f"""
IMPORTANT: Previous round {round_num + 1} made things WORSE.
- Attempted rewrites of {len(best_result['fake_sentences'])} sentences
- Result: {best_result['score']:.1f}% (increased from {best_ever_score:.1f}%)
- New sentences got flagged that weren't before

These rewrites FAILED and made detection worse. Learn from this.
DO NOT repeat similar patterns. Try fundamentally different approaches.
"""
				
				round_num += 1
				continue  # Try again with rolled-back state
			
			# Update current state
			current_text = best_candidate["text"]
			previous_score = best_result["score"]
			current_fake_sentences = best_result["fake_sentences"]
			
			print(f"\nRound result: {previous_score:.1f}% ({len(current_fake_sentences)} flagged)")
			
			round_num += 1
		
		# At the end, use best_ever_text, not current_text
		print(f"\n{'='*50}")
		print(f"Final score: {best_ever_score:.1f}% (from round {best_ever_round})")
		
		print("\nScore progression:")
		for entry in score_history:
			status = "✓" if entry["accepted"] else "↩" if entry["rolled_back"] else "○"
			print(f"  Round {entry['round']}: {entry['best_candidate_score']:.1f}% {status}")
		
		print(f"\nFinal Result:\n{best_ever_text}")
	
	except Exception as e:
		print(f"\n✗ Fatal error: {e}")
		raise
	finally:
		# Cleanup
		await checker.close()

if __name__ == "__main__":
	asyncio.run(main())