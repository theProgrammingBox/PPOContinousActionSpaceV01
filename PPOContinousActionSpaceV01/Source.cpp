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
    olc::vf2d policies[arrSize];
    float values[arrSize];
    float actions[arrSize];
    float rewards[arrSize];
    float logProbabilities[arrSize];
    float discountedRewards[arrSize];
    float advantages[arrSize];
    float valueGradients[arrSize];
    olc::vf2d policyGradients[arrSize];

    olc::vf2d* observationPtr;
    olc::vf2d* policyPtr;
    float* valuePtr;
    float* actionPtr;
    float* rewardPtr;
    float* logProbabilityPtr;
    float* discountedRewardPtr;
    float* advantagePtr;
    float* valueGradientPtr;
    olc::vf2d* policyGradientPtr;

    float tmp;
    float lastDiscountedReward;
    float lastAdvantage;
    float lastValue;
    float klDivergence;
    olc::vf2d updatedPolicy;
    float updatedValue;
    float updatedAction;
    float newLogProb;
    float ratio;
    float clipRatio;

    for (uint32_t epoch = maxEpoch; epoch--;)
    {
        observationPtr = observations;
        policyPtr = policies;
        valuePtr = values;
        actionPtr = actions;
        logProbabilityPtr = logProbabilities;
        rewardPtr = rewards;
        for (uint32_t rollout = maxRollouts; rollout--;)
        {
            env.reset(observationPtr);
            for (uint32_t step = maxGameSteps; step--;)
            {
                nn.forward(observationPtr, policyPtr, valuePtr);
                nn.sample(policyPtr, actionPtr);
                tmp = (*actionPtr - (*policyPtr).x) / (*policyPtr).y;
                *logProbabilityPtr = -0.5f * tmp * tmp - log((*policyPtr).y) - 0.9189385332046727f;
                
                observationPtr++;
                
                env.step(actionPtr, observationPtr, rewardPtr);

                policyPtr++;
                valuePtr++;
                actionPtr++;
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
            valueGradientPtr = valueGradients;
            discountedRewardPtr = discountedRewards;
            logProbabilityPtr = logProbabilities;
            policyGradientPtr = policyGradients;
            advantagePtr = advantages;
            for (uint32_t rollout = maxRollouts; rollout--;)
            {
                for (uint32_t step = maxGameSteps; step--;)
                {
                    nn.forward(observationPtr, &updatedPolicy, &updatedValue);
                    nn.sample(&updatedPolicy, &updatedAction);
                    *valueGradientPtr = 2 * (*discountedRewardPtr - updatedValue);
                    tmp = (updatedAction - updatedPolicy.x) / updatedPolicy.y;
                    newLogProb = -0.5f * tmp * tmp - log(updatedPolicy.y) - 0.9189385332046727f;
                    
                    ratio = exp(newLogProb - *logProbabilityPtr);
                    clipRatio = std::min(std::max(ratio, lowerBound), upperBound);
                    *policyGradientPtr = olc::vf2d(tmp / updatedPolicy.y, (tmp * tmp - 1.0f) / updatedPolicy.y) * *advantagePtr * std::min(ratio, clipRatio);
                    klDivergence += *logProbabilityPtr - newLogProb;

                    observationPtr++;
                    valueGradientPtr++;
                    discountedRewardPtr++;
                    logProbabilityPtr++;
                    policyGradientPtr++;
                    advantagePtr++;
                }
            }
            if (klDivergence / arrSize > klThreshold)
                break;
        }
    }

    return 0;
}
