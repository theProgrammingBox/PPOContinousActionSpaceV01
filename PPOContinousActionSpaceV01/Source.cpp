#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <random>

/*
High Level Overview:
1. collect N full games. each step in the game should store the observation, updatedAction, updatedPolicy log probabilities, updatedValue, reward
2. use a full rollout to calculate the returns and advantage for each step
3. use the returns to calculate the updatedValue loss
4. use the advantage, old updatedPolicy log probabilities, and new updatedPolicy log probabilities to calculate the updatedPolicy loss
*/

/*
TODO:
0. remove policyPtr and arr if not needed
1. see if there is a need for the ptr vars. can remove otherize cuz messy
*/

struct Environment
{
    void reset(olc::vf2d* observation)
    {
	}

    void step(float* updatedAction, olc::vf2d* observation, float* reward)
    {
	}
};

struct NeuralNetwork
{
    void forward(olc::vf2d* observation, olc::vf2d* updatedPolicy, float* updatedValue)
    {
	}

    void sample(olc::vf2d* updatedPolicy, float* updatedAction)
    {
	}
};

int main()
{
    const uint32_t maxEpoch = 100000;
    const uint32_t maxUpdates = 16;
    const uint32_t maxRollouts = 16;
    const uint32_t maxGameSteps = 16;
    const uint32_t arrSize = maxRollouts * maxGameSteps;

    const float invArrSize = 1.0f / arrSize;
    const float discountFactor = 0.99f;
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
    float klDivergence;
    olc::vf2d updatedPolicy;
    float updatedValue;
    float updatedAction;

    for (uint32_t epoch = 0; epoch < maxEpoch; epoch++)
    {
        observationPtr = observations;
        policyPtr = policies;
        valuePtr = values;
        actionPtr = actions;
        rewardPtr = rewards;
        logProbabilityPtr = logProbabilities;
        for (uint32_t rollout = 0; rollout < maxRollouts; rollout++)
        {
            env.reset(observationPtr);
            for (uint32_t step = 0; step < maxGameSteps; step++)
            {
                nn.forward(observationPtr, policyPtr, valuePtr);
                nn.sample(policyPtr, actionPtr);
                // this iteration is a problem
                observationPtr++;
                env.step(actionPtr, observationPtr, rewardPtr);
                tmp = (*actionPtr - (*policyPtr).x) / (*policyPtr).y;
                *logProbabilityPtr = -0.5f * tmp * tmp - log((*policyPtr).y) - 0.9189385332046727f;

                valuePtr++;
                policyPtr++;
                actionPtr++;
                rewardPtr++;
                logProbabilityPtr++;
            }
        }

        // place holder for discountedRewards and advantages math
        // see if we can reverse the entire thing to use ptrs
        for (uint32_t rollout = maxRollouts; rollout--;)
        {
            float discountedReward = rewards[rollout * maxGameSteps + maxGameSteps - 1];
            discountedRewards[rollout * maxGameSteps + maxGameSteps - 1] = discountedReward;
            float lastAdvantage = rewards[rollout * maxGameSteps + maxGameSteps - 1] - values[rollout * maxGameSteps + maxGameSteps - 1];
            advantages[rollout * maxGameSteps + maxGameSteps - 1] = lastAdvantage;
            for (uint32_t step = maxGameSteps - 1; step--;)
            {
                discountedReward = rewards[rollout * maxGameSteps + maxGameSteps] + discountFactor * discountedReward;
                discountedRewards[rollout * maxGameSteps + maxGameSteps] = discountedReward;
                lastAdvantage = rewards[rollout * maxGameSteps + maxGameSteps] + discountFactor * values[rollout * maxGameSteps + maxGameSteps - 1]
                advantages[]
            }
        }

        for (uint32_t iteration = 0; iteration < maxUpdates; iteration++)
        {
            klDivergence = 0;
            observationPtr = observations;
            valueGradientPtr = valueGradients;
            discountedRewardPtr = discountedRewards;
            logProbabilityPtr = logProbabilities;
            policyGradientPtr = policyGradients;
            advantagePtr = advantages;
            for (uint32_t rollout = 0; rollout < maxRollouts; rollout++)
            {
                for (uint32_t step = 0; step < maxGameSteps; step++)
                {
                    nn.forward(observationPtr, &updatedPolicy, &updatedValue);
                    nn.sample(&updatedPolicy, &updatedAction);
                    *valueGradientPtr = 2 * (*discountedRewardPtr - updatedValue);
                    tmp = (updatedAction - updatedPolicy.x) / updatedPolicy.y;
                    float newLogProb = -0.5f * tmp * tmp - log(updatedPolicy.y) - 0.9189385332046727f;
                    
                    float ratio = exp(newLogProb - *logProbabilityPtr);
                    float clipRatio = std::min(std::max(ratio, lowerBound), upperBound);
                    *policyGradientPtr = olc::vf2d(0, 0) * std::min(*advantagePtr * ratio, *advantagePtr * clipRatio);
                    klDivergence += *logProbabilityPtr - newLogProb;

                    observationPtr++;
                    valueGradientPtr++;
                    discountedRewardPtr++;
                    logProbabilityPtr++;
                    policyGradientPtr++;
                    advantagePtr++;
                }
            }
            if (klDivergence * invArrSize > klThreshold)
                break;
        }
    }

    return 0;
}
