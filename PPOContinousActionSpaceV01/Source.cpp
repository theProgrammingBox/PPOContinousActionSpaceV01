#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <random>

// Include necessary libraries (cuBLAS, cuDNN, cuRAND)

float randomFloat(float min, float max)
{
    static std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(min, max);
    return distribution(generator);
}

class Environment
{
private:
    const olc::vf2d lowerBound = { 0.0f, 0.0f };
    const olc::vf2d upperBound = { 1000.0f, 500.0f };
    const float sqrTargetRadius = 100.0;
    const int maxSteps = 16;

    int currentStep;
    olc::vf2d agentPosition;
    olc::vf2d targetPosition;

    olc::vf2d randomVector(const olc::vf2d& min, const olc::vf2d& max)
    {
		return { randomFloat(min.x, max.x), randomFloat(min.y, max.y) };
	}

    void GetObservation(olc::vf2d& observation)
    {
        observation = agentPosition - targetPosition;
	}

public:
    void reset(olc::vf2d& observation)
    {
        currentStep = 0;
        agentPosition = randomVector(lowerBound, upperBound);
        targetPosition = randomVector(lowerBound, upperBound);
        GetObservation(observation);
    }

    void step(const olc::vf2d& action, olc::vf2d& observation, float& reward, bool& notDone)
    {
        currentStep++;
        agentPosition += action;
        agentPosition.clamp(lowerBound, upperBound);

        reward = 0;
        if ((agentPosition - targetPosition).mag2() <= sqrTargetRadius)
        {
            targetPosition = randomVector(lowerBound, upperBound);
            reward = 1;
        }
        notDone = currentStep < maxSteps;
        GetObservation(observation);
    }
};


class NeuralNetwork
{
public:
    void forward(const olc::vf2d& observation, std::tuple<std::tuple<float, float>, std::tuple<float, float>>& policy, float& value)
    {
        policy = std::make_tuple(std::make_tuple(randomFloat(-1, 1), randomFloat(-1, 1)), std::make_tuple(randomFloat(-1, 1), randomFloat(-1, 1)));
        value = randomFloat(-1, 1);
    }
    void sample(const std::tuple<std::tuple<float, float>, std::tuple<float, float>>& policy, olc::vf2d& action)
    {
        action = { randomFloat(-1, 1), randomFloat(-1, 1) };
    }
};

struct RolloutData
{
    olc::vf2d observation;
    olc::vf2d action;
    std::tuple<std::tuple<float, float>, std::tuple<float, float>> policy;
    float value;
    float reward;
    bool notDone;
    float return_;
    float advantage;
};

class PPO
{
private:
    const float discount = 0.99;
    const float gamma = 0.99;
    const float lambda = 0.95;
public:
    PPO(Environment& env, NeuralNetwork& nn)
    {
        std::vector<std::vector<RolloutData>> rollouts;
        for (uint32_t i = 0; i < 1; ++i)
        {
            std::vector<RolloutData> rollout;

            olc::vf2d observation;
            std::tuple<std::tuple<float, float>, std::tuple<float, float>> policy;
            float value;
            olc::vf2d action;
            float reward;
            bool notDone = true;
            float return_ = 0;
            float advantage = 0;
            env.reset(observation);

            while (notDone)
            {
                nn.forward(observation, policy, value);
                nn.sample(policy, action);
                env.step(action, observation, reward, notDone);

                rollout.push_back({ observation, action, policy, value, reward, notDone, return_, advantage });
			}

            return_ = value;
            rollout.back().return_ = return_;

            float lastAdvantage = rollout[i].reward - rollout[i].value;
            rollout.back().advantage = lastAdvantage;
            float lastValue = value;

            for (int i = rollout.size() - 2; i >= 0; --i)
            {
				return_ = rollout[i].reward + discount * return_;
				rollout[i].return_ = return_;

                lastAdvantage = (rollout[i].reward + gamma * lastValue - rollout[i].value) + gamma * lambda * lastAdvantage;
                rollout[i].advantage = lastAdvantage;
                lastValue = rollout[i].value;
			}
			rollouts.push_back(rollout);

            //print out rollout data
            for (uint32_t i = 0; i < rollouts.size(); ++i)
            {
                for (uint32_t j = 0; j < rollouts[i].size(); ++j)
                {
					std::cout << "Observation: " << rollouts[i][j].observation << std::endl;
					std::cout << "Action: " << rollouts[i][j].action << std::endl;
                    // for policy, need to unpack tuple into 2 means and 2 stds
                    std::cout << "Policy x mean: " << std::get<0>(std::get<0>(rollouts[i][j].policy)) << std::endl;
                    std::cout << "Policy x std: " << std::get<1>(std::get<0>(rollouts[i][j].policy)) << std::endl;
                    std::cout << "Policy y mean: " << std::get<0>(std::get<1>(rollouts[i][j].policy)) << std::endl;
                    std::cout << "Policy y std: " << std::get<1>(std::get<1>(rollouts[i][j].policy)) << std::endl;
					std::cout << "Value: " << rollouts[i][j].value << std::endl;
					std::cout << "Reward: " << rollouts[i][j].reward << std::endl;
					std::cout << "Not Done: " << rollouts[i][j].notDone << std::endl;
					std::cout << "Return: " << rollouts[i][j].return_ << std::endl;
					std::cout << "Advantage: " << rollouts[i][j].advantage << std::endl;
					std::cout << std::endl;
				}
			}
        }
    }
};

int main()
{
    Environment env;
    NeuralNetwork nn;
    PPO ppo(env, nn);
}
