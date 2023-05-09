#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <random>

/*
TODO:
0. remove policyPtr and arr if not needed
1. see if there is a need for the ptr vars. can remove otherize cuz messy
2. make sure env doesn't alter observation when game over due to how the loop works
3. alter the buffers so they can handle dynamic lengths
*/

struct Environment
{
    void reset(olc::vf2d* observation)
    {
		*observation = { 0.0f, 0.0f };
	}

    void step(float* action, olc::vf2d* observation, float* reward)
    {
        if (true)
        {
            *reward = 0.0f;
        }
        if (true)
        {
            *observation = { 0.0f, 0.0f };
        }
	}
};

struct NeuralNetwork
{
    void forward(olc::vf2d* observation, olc::vf2d* policy, float* value)
    {
        *policy = { 0.0f, 0.0f };
        *value = 0.0f;
    }

    void sample(olc::vf2d* policy, float* action)
    {
		*action = 0.0f;
	}
};

int main()
{
    const uint32_t maxEpoch = 100000;
    const uint32_t maxUpdates = 16;
    const uint32_t maxRollouts = 16;
    const uint32_t maxGameSteps = 16;
    const uint32_t arrSize = maxRollouts * maxGameSteps;
    
    const float discountFactor = 0.99f;
	const float lambda = 0.95f;
    const float epsilon = 0.2f;
    const float upperBound = 1.0f + epsilon;
    const float lowerBound = 1.0f - epsilon;
    const float klThreshold = 0.01f;

    Environment env;
    NeuralNetwork nn;

    olc::vf2d observations[arrSize];
    float values[arrSize];
    float rewards[arrSize];
    float logProbabilities[arrSize];
    float discountedRewards[arrSize];
    float advantages[arrSize];
    olc::vf2d policyGrads[arrSize];
    float valueGrads[arrSize];

    olc::vf2d* observationPtr;
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
    float action;
    float logProb;
    float klDivergence;
    float ratio;
    float clipRatio;

    for (uint32_t epoch = maxEpoch; epoch--;)
    {
        observationPtr = observations;
        valuePtr = values;
        logProbabilityPtr = logProbabilities;
        rewardPtr = rewards;
        for (uint32_t rollout = maxRollouts; rollout--;)
        {
            env.reset(observationPtr);
            for (uint32_t step = maxGameSteps; step--;)
            {
                nn.forward(observationPtr, &policy, valuePtr);
                nn.sample(&policy, &action);
                tmp = (action - policy.x) / policy.y;
                *logProbabilityPtr = -0.5f * tmp * tmp - log(policy.y) - 0.9189385332046727f;
                
                observationPtr++;
                
                env.step(&action, observationPtr, rewardPtr);
                
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
            klDivergence = 0;
            observationPtr = observations;
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
                    nn.sample(&policy, &action);
                    
                    tmp = 1.0f / policy.y;
                    policyGradPtr->x = (action - policy.x) * tmp * tmp;
                    policyGradPtr->y = policyGradPtr->x * policyGradPtr->x * policy.y - tmp;
                    logProb = -0.5f * policyGradPtr->y * policy.y - log(policy.y) - 1.4189385332046727f;
                    
                    klDivergence += *logProbabilityPtr - logProb;
                    
                    ratio = exp(logProb - *logProbabilityPtr);
                    clipRatio = std::min(std::max(ratio, lowerBound), upperBound);
                    *policyGradPtr *= std::min(*advantagePtr * ratio, *advantagePtr * clipRatio);

                    *valueGradPtr = 2 * (*discountedRewardPtr - value);

                    observationPtr++;
                    discountedRewardPtr++;
                    logProbabilityPtr++;
                    policyGradPtr++;
                    advantagePtr++;
                    valueGradPtr++;
                }
            }
            if (klDivergence / arrSize > klThreshold)
                break;
            // update model
        }
    }

    return 0;
}
