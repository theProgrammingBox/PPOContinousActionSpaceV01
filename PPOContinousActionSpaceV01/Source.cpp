#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <random>

/*
Important lessons:
0. The value function is basically the average discounted reward expected to be received
from a given state given that the agent follows the policy distribution.
1. PPO works by having the policy distribution sample moves which all result in different
discounted rewards. Then, we look at the value, aka the average reward expected and the
reward we got from the move. If the reward is higher than the average, we want to increase
the probability of the move, and vice versa. That is basically what advantage is.
3. the important thing about ppo is the clipping. When we update the policy, we check if that
same move is more/less likely to be taken. we do this by taking a ratio of their log probabilities.
if the ratio is greater or less than 1 + epsilon, we clip it to 1 + epsilon or 1 - epsilon
respectively. I still dont fully understand why this works, but it does.
4. Additionally, I included the kl divergence as a way to check if the policy is changing too much.
I plugged in the equation on desmos and I can kind of see how it works, but I still don't fully
understand it.
*/

/*
TODO:
0. implement a basic test where the outouts and values are constant, just train a single step
1. test the current implementation to see if it works
2. make sure env doesn't alter observation when game over due to how the loop works
3. alter the buffers so they can handle dynamic lengths
*/

struct Environment
{
	void reset(olc::vf2d* observationReturn)
	{
		*observationReturn = { 0.0f, 0.0f };
	}

	void step(float* actionInput, olc::vf2d* observationReturn, float* rewardReturn)
	{
		*rewardReturn = std::min(0.0f, 2.0f - abs(-9.0f - *actionInput));
		if (false)
		{
			*observationReturn = { 0.0f, 0.0f };
		}
	}
};

struct NeuralNetwork
{
	olc::vf2d policy = { 0.0f, 1.0f };
	float value = 0;

	void forward(olc::vf2d* observationInput, olc::vf2d* policyReturn, float* valueReturn)
	{
		*policyReturn = policy;
		*valueReturn = value;
	}

	void sample(olc::vf2d* policyInput, float* actionReturn)
	{
		static std::random_device rd;
		static std::mt19937 gen(rd());
		static std::normal_distribution<float> d(0, 1);
		*actionReturn = policyInput->x + policyInput->y * d(gen);
	}

	void update(olc::vf2d* policyGradInput, float* valueGradInput, float policyLearningRateInput, float valueLearningRateInput)
	{
		policy += *policyGradInput * policyLearningRateInput;
		policy.y = std::max(0.1f, policy.y);
		value += *valueGradInput * valueLearningRateInput;
	}
};

int main()
{
	const uint32_t maxEpoch = 512;
	const uint32_t maxUpdates = 16;
	const uint32_t maxRollouts = 16;
	const uint32_t maxGameSteps = 1;
	const uint32_t arrSize = maxRollouts * maxGameSteps;

	const float discountFactor = 0.99f;
	const float lambda = 0.95f;
	const float epsilon = 0.2f;
	const float upperBound = 1.0f + epsilon;
	const float lowerBound = 1.0f - epsilon;
	const float klThreshold = 0.02f;
	const float policyLearningRate = 0.004f / arrSize;
	const float valueLearningRate = 0.04f / arrSize;

	Environment env;
	NeuralNetwork nn;

	olc::vf2d observations[arrSize];
	float actions[arrSize];
	float values[arrSize];
	float rewards[arrSize];
	float logProbabilities[arrSize];
	float discountedRewards[arrSize];
	float advantages[arrSize];
	olc::vf2d policyGrads[arrSize];
	float valueGrads[arrSize];

	olc::vf2d* observationPtr;
	float* actionPtr;
	float* valuePtr;
	float* rewardPtr;
	float* logProbabilityPtr;
	float* discountedRewardPtr;
	float* advantagePtr;
	olc::vf2d* policyGradPtr;
	float* valueGradPtr;

	float tmp;
	float lastDiscountedReward;
	float lastAdvantage;
	float lastValue;
	olc::vf2d policy;
	float value;
	float logProb;
	//float klDivergence;
	float ratio;
	float clipRatio;
	float policyLoss;

	for (uint32_t epoch = maxEpoch; epoch--;)
	{
		observationPtr = observations;
		actionPtr = actions;
		valuePtr = values;
		logProbabilityPtr = logProbabilities;
		rewardPtr = rewards;
		for (uint32_t rollout = maxRollouts; rollout--;)
		{
			env.reset(observationPtr);
			for (uint32_t step = maxGameSteps; step--;)
			{
				nn.forward(observationPtr, &policy, valuePtr);
				nn.sample(&policy, actionPtr);
				tmp = (*actionPtr - policy.x) / policy.y;
				*logProbabilityPtr = -0.5f * tmp * tmp - log(policy.y) - 0.9189385332046727f;

				observationPtr++;

				env.step(actionPtr, observationPtr, rewardPtr);

				actionPtr++;
				valuePtr++;
				logProbabilityPtr++;
				rewardPtr++;
			}
		}

		rewardPtr = rewards + arrSize - 1;
		discountedRewardPtr = discountedRewards + arrSize - 1;
		valuePtr = values + arrSize - 1;
		advantagePtr = advantages + arrSize - 1;
		for (uint32_t rollout = maxRollouts; rollout--;)
		{
			lastDiscountedReward = 0;
			lastAdvantage = 0;
			lastValue = 0;
			for (uint32_t step = maxGameSteps; step--;)
			{
				lastDiscountedReward = *rewardPtr + discountFactor * lastDiscountedReward;
				*discountedRewardPtr = lastDiscountedReward;

				lastAdvantage = *rewardPtr + discountFactor * lastValue - *valuePtr + discountFactor * lambda * lastAdvantage;
				lastValue = *valuePtr;
				*advantagePtr = lastAdvantage;

				rewardPtr--;
				discountedRewardPtr--;
				valuePtr--;
				advantagePtr--;
			}
		}

		for (uint32_t iteration = maxUpdates; iteration--;)
		{
			//klDivergence = 0;
			policyLoss = 0;
			observationPtr = observations;
			actionPtr = actions;
			discountedRewardPtr = discountedRewards;
			logProbabilityPtr = logProbabilities;
			policyGradPtr = policyGrads;
			advantagePtr = advantages;
			valueGradPtr = valueGrads;
			for (uint32_t rollout = maxRollouts; rollout--;)
			{
				for (uint32_t step = maxGameSteps; step--;)
				{
					nn.forward(observationPtr, &policy, &value);

					tmp = 1.0f / policy.y;
					policyGradPtr->x = (*actionPtr - policy.x) * tmp * tmp;
					policyGradPtr->y = policyGradPtr->x * policyGradPtr->x * policy.y - tmp;
					logProb = -0.5f * policyGradPtr->y * policy.y - log(policy.y) - 1.4189385332046727f;

					//klDivergence -= exp(*logProbabilityPtr) * (*logProbabilityPtr - logProb);

					ratio = exp(logProb - *logProbabilityPtr);
					clipRatio = std::min(std::max(ratio, lowerBound), upperBound);
					tmp = std::min(*advantagePtr * ratio, *advantagePtr * clipRatio);
					*policyGradPtr *= tmp;
					policyLoss += abs(tmp);

					*valueGradPtr = 2 * (*discountedRewardPtr - value);

					observationPtr++;
					actionPtr++;
					discountedRewardPtr++;
					logProbabilityPtr++;
					policyGradPtr++;
					advantagePtr++;
					valueGradPtr++;
				}
			}

			/*if (klDivergence / arrSize > klThreshold)
				break;*/
			//printf("klDivergence: %f\n", klDivergence / arrSize);

			policyGradPtr = policyGrads;
			valueGradPtr = valueGrads;
			for (uint32_t rollout = maxRollouts; rollout--;)
			{
				for (uint32_t step = maxGameSteps; step--;)
				{
					nn.update(policyGradPtr, valueGradPtr, policyLearningRate, valueLearningRate);

					policyGradPtr++;
					valueGradPtr++;
				}
			}
		}
		/*printf("policy3: %f, %f\n", nn.policy.x, nn.policy.y);
		printf("value: %f\n", nn.value);*/
		printf("policyLoss: %f\n", policyLoss / arrSize);
	}

	return 0;
}
