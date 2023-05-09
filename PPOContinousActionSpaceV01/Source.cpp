#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <random>

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
        *rewardReturn = -abs(10.0f - *actionInput);
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

    void update(olc::vf2d* policyGradInput, float* valueGradInput, float learningRateInput)
    {
		policy += *policyGradInput * learningRateInput;
        value += *valueGradInput * learningRateInput;
    }
};

int main()
{
    const uint32_t maxEpoch = 10;
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
    const float learningRate = 0.001f / arrSize;

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
    float valueLoss;
    float policyLoss;

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

        printf("\nh\n");
        for (uint32_t iteration = maxUpdates; iteration--;)
        {
            klDivergence = 0;
            valueLoss = 0;
            policyLoss = 0;
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
                    
                    // unoptimize the function to get better accuracy
                    tmp = 1.0f / policy.y;
                    policyGradPtr->x = (action - policy.x) * tmp * tmp;
                    policyGradPtr->y = policyGradPtr->x * policyGradPtr->x * policy.y - tmp;
                    logProb = -0.5f * policyGradPtr->y * policy.y - log(policy.y) - 1.4189385332046727f;
                    
                    klDivergence += *logProbabilityPtr - logProb;
                    
                    ratio = exp(logProb - *logProbabilityPtr);
                    clipRatio = std::min(std::max(ratio, lowerBound), upperBound);
                    tmp = std::min(*advantagePtr * ratio, *advantagePtr * clipRatio);
                    *policyGradPtr *= tmp;
                    policyLoss -= tmp;

                    tmp = *discountedRewardPtr - value;
                    *valueGradPtr = 2 * tmp;
                    valueLoss += tmp * tmp;

                    observationPtr++;
                    discountedRewardPtr++;
                    logProbabilityPtr++;
                    policyGradPtr++;
                    advantagePtr++;
                    valueGradPtr++;
                }
            }/**/

            // print stats
            if (iteration + 1 == maxUpdates)
            {
                printf("valueLoss: %f\n", valueLoss / arrSize);
                printf("policyLoss: %f\n", policyLoss / arrSize);
                printf("klDivergence: %f\n", klDivergence / arrSize);
                printf("policy: %f, %f\n", policy.x, policy.y);
                printf("value: %f\n", value);
                printf("action: %f\n", action);
            }
            
            if (klDivergence / arrSize > klThreshold)
                break;
            
            // update model
            /*policyGradPtr = policyGrads;
            valueGradPtr = valueGrads;
            for (uint32_t rollout = maxRollouts; rollout--;)
            {
                for (uint32_t step = maxGameSteps; step--;)
                {
                    nn.update(policyGradPtr, valueGradPtr, learningRate);

                    policyGradPtr++;
                    valueGradPtr++;
                }
            }*/
        }
    }

    return 0;
}
