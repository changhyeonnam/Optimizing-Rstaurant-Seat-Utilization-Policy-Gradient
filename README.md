# Optimizing Restaurant seat utilization with Policy-based Reinforcement Learning (ORSU)

# 1. Abstract

인하대 후문 근처 식당가에서는 점심시간에 혼밥(’혼자 밥먹기’의 줄임말)학생들을 받지 않는 경우가 더러 있다. 왜 식당들은 점심시간에 혼자오는 손님들을 거부할까? 단순히 고정관념에 의한 의사결정인지 실제 데이터를 기반으로한 전략적 의사결정인지 불분명했고, 전략적 의사결정이라면 혼밥손님을 받았을 때와 그렇지 않았을 때 경제적 이득 차이가 어느 정도 나는지 알아보고 싶었다.

그래서 우리는 1인 손님과 단체 손님으로 이루어진 집합 내에서 식당 측의 경제적 이득을 최적화 할 수 있는 스케줄링 방법, 자리 배치 방법을 알아보고자 한다.

관습적으로 기존 식당에서는 1인 손님을 받는 것은 4인 테이블의 공간을 낭비한다는 근거 하에 단체 손님 위주로 받아왔다. 또는 단순 휴리스틱한 방법으로 몇가지 경우만을 고려하여, 단체 손님 위주로 스케줄링/자리 배치 하는 전략을 사용해왔다.

우리는 식당 측에서 가장 많은 경제적 이득을 낼 수 있는 자리 배치와 손님 스케줄링 방법을 찾기 위해 강화학습을 적용 하였다. Real world의 식당에서의 손님 분포를 고려하여, 몇명의 손님인지, 순서는 어떻게 되는지, 비용은 얼마나 걸리는지 등을 가정하였다. 먼저, 강화학습을 적용하기 전에 해야할 몇가지 가정을 설명하고, Markov Decision Process와 관련한 setting을 설명하겠다. 그 다음 RL 알고리즘을 어떻게 적용했고, 어떤 결과를 기대하는지 설명하겠다.

( 현재 환경에 대해 시뮬레이션 하는 코드까지 완료된 상태이면, 추후 강화학습을 적용해보려한다. [https://github.com/changhyeonnam/Optimizing-Rstaurant-Seat-Utilization-Policy-Gradient](https://github.com/changhyeonnam/Optimizing-Rstaurant-Seat-Utilization-Policy-Gradient))


# 2. Assumptions

State, Action, Reward를 정하기 전에 Real world를 고려한 손님과 자리 배치에 대한 가정 몇가지를 해보겠다.

1. 손님의 인원수는 최소 1인, 최대 4인으로 한정하였다.
2. 인원 수에 따른 손님의 식사 시간 : 1인 손님이 식사할때와 단체 손님이 식사할때 소모되는 시간이 다르다. 1인 손님의 식사시간이 t라고 했을 때, 다음과 같이 손님에 대한 식사시간을 가정하였다.


    |  | 1인 | 2인 | 3인 | 4인 |
    | --- | --- | --- | --- | --- |
    | Time | t | 1.5t | 2.25t | 3.375t |

    실험의 편의를 위해 t=1이라고 가정하였다.

3. 인원 수에 따른 손님의 주문 비용: 손님 1명당 주문하는 비용은 b로 일정하다고 가정한다.


    |  | 1인 | 2인 | 3인 | 4인 |
    | --- | --- | --- | --- | --- |
    | Cost | b | 2b | 3b | 4b |
4. 인원 수에 따른 손님의 분포 : 먼저, 총 인원 수가 50명이라고 가정하였다. 50명에 대해 3가지 케이스에 대해 먼저 실험을 해보려 한다.  
    1. case 1: 1인 손님이 가장 많은 경우
    2. case 2: 2인 손님이 가장 많은 경우
    3. case 3: 3인,4인이 손님이 가장 경우

    |  | 1인  | 2인  | 3인 | 4인 | total |
    | --- | --- | --- | --- | --- | --- |
    | case 1 | 16 | 3 | 4 | 4 | 50 |
    | case 2 | 6 | 8 | 4 | 4 | 50 |
    | case 3 | 2 | 3 | 6 | 6 | 50 |
5. 인원 수에 따른 손님의 자리 배치 모양 : 좌석 배치를 위해 어떤 형태로 손님이 앉아야 하는지 정의해야한다. 총 5가지의 형태로 정의하였다.


    1. 1인 손님 : 1가지

        ![Untitled](https://i.imgur.com/b3s4Xxf.png)

    2. 2인 손님 : 2가지

        ![Untitled 1](https://i.imgur.com/QmQmLLh.png)

    3. 3인 손님 : 1가지. 실제 세계에서는 4인 좌석에 3명이 앉는 경우가 대다수이기 때문에 아래와 같이 정의하였다. 네개의 칸 중 실제로는 한칸이 비여있다.

        ![Untitled 2](https://i.imgur.com/6yQVRnz.png)
    4. 4인 손님 : 1가지

        ![Untitled 3](https://i.imgur.com/npcbS6k.png)

6. 식당의 좌석 구조 : 기본적으로 5x5 grid형태의 좌석을 사용하였다.

![Untitled 4](https://i.imgur.com/fP3bBk5.png)

실제 세계에 적용하기 위해서는 사용하지 못하는 구역을 지정해주면 될 것같다. 현재는 가장 단순하게 5x5 grid를 모두 사용하는 방법으로 결정하였다.


# 3. Method

우리가 하고자 하는 것은 손님 n팀으로 이루어져 있는 waiting list가 있을때, 매 timestep마다 가장 경제적 이득을 많이 주는 1팀을 선정하여 자리배치를 하는 것이다. 최대 n개의 팀이 waiting list에 있고, waiting list는 매 timestep마다 한팀씩 들어온다고 가정하였다. (첫 timestep에서는 3팀이 기다리고 있다고 가정.) 손님 선정 스케줄러에 의해 어떤 손님이 가게에서 먹을지 정해지면, 좌석 배치는 **너비 우선 탐색**(Breadth-first search, **BFS**)방식으로 좌표(0,0)부터 탐색하여 가능한 자리에 배치되게 된다. 좌석에 앉은 손님의 식사시간이 모두 종료되면, 빈 좌석이 된다. Episode의 종료 시점은 모든 손님이 스케줄링 되어 식사를 끝 마쳤을 때이다.

## 3.1 Markov Decision Property Settings

### State

좌석에 배치되어 식사를 하고 있는 손님과 n명이 기다리고 있는 waiting list에 대한 discreste image로 현재 state를 표현하였다. 실제 프로그램에서는 손님마다 다른 색을 사용하여 각 손님을 식별 가능하게 하였다.

![Untitled 5](https://i.imgur.com/c89lmCx.png)

위의 형태와 같은 Fixed state representation으로 State를 표현하게 되면, Convoluitional Neural Network(CNN)의 입력으로 넣을 수 있게된다.

### Action

각 timstep마다, 스케줄러는 n개의 팀으로 이루어져 있는 waiting list에서 어떤 손님들을 고를지 결정해야한다. 한번에 1팀 이상의 손님을 배치하기 위해서는 $O(2^n)$의 경우를 고려한 action이 필요하다. 하지만 이 방법은 학습을 어렵하게 만들다. 이를 위해서, Resource Management with Deep Reinforcement Learning의 논문에서 사용한 trick을 사용하였다. Action은 {$\empty$,1,…,n}으로 주어지고, a = i가 의미하는 것은 waiting list의 i-th team을 좌석 배치하라는 의미이다. 공집합은 현재 agent가 좌석배치하지 않는 다는 뜻이다.

만약 유효한 action을 하게 되면, 손님이 좌석에 배치되게 되고, Agent는 그 action에 따른 state를 관찰할 수 있게 된다. 만약 공집합 혹은 유효하지 않은 action(좌석이 모두 가득찬 경우)을 하게되면, 현재 state와 동일한 state를 관찰하게 된다.

### Reward

총 두가지의 요소로 Reward를 정의 하였다.

1. 단위 시간당 들어오는 수익(Profit, P라고 표현).

    $n_{t,i}$ : 현재 timestep에서 식사 하고 있는 i명의 단체손님의 개수.

    $$P_t={\sum_{i=1}^4 i *{b_i\over t_i}*n_{t,i}}$$

2. 단위 시간당 빈자리로 인한 손해 (Cost, C라고 표현)

    먼저 한자리 당 나올 수 있는 손해에 대한 expecation 값을 아래와 같이 미리 계산하고, 빈자리 개수를 곱해주면 구할 수 있다.

    $b_i$ : i명의 단체손님의 미리 가정된 수익

    $t_i$ : i명의 단체 손님의 식사 시간

    $n_i$ : i명의 단체손님의 개수

    $e_t$ : 현재 timestep t에서의 빈자리 개수

$$ C_t = \frac {\sum_{i=1}^4i*{b_i\over t_i}*n_i} {\sum_{i=1}^4 i* n_i *e_t} $$


위에서 정의한 $R_t,C_t$를 이용하여 다음과 같이 보상함수를 정의 하였다.

$$r_t(s_t,a_t,s'_t) = P_t-0.5*C_t$$

이때 단위시간당 들어오는 수익을 손해보다 좀더 우선시 하기 위해, Cost에 0.5를 곱해주었다. 우리가 원하는 누적 경제적이득을 최대화 하기 위해, Discount Factor $\gamma=1$로 가정하였다.

### Training Algorithm

Deep Learning을 사용하는 강화학습 알고리즘 중 하나인 REINFORCE 알고리즘을 사용하였다. 기본적으로 episodic setting에서 강화학습을 적용하였다. 매 episode마다 n개의 손님으로 이루어져 있는 팀이 들어오고, policy에 의해 어떤 팀들이 식당에 배치될지 정해지고, 모든 손님이 식사를 끝 맞추면 Episode가 종료되게 된다. REINFOCE 알고리즘의 대표적인 단점은 graident estimate에 있어서 높은 variance가 있다는 것이다. 이를 위해 $v_t$에 baseline값을 빼주는 방식으로 사용하는 방법이 자주 사용되어, 우리도 이 방법을 적용하였다. 다음은 pseudo code이다.

![Screen Shot 2023-01-25 at 2.51.56 PM](https://i.imgur.com/XPHyirXl.png)

앞서 언급한 Gradient estiamte에서의 high variance 문제를 해결한 Proximal Policy Optimization을 적용한다면 보다 좋은 성능의 알고리즘을 설계할 수 있을 것이라 생각한다.


# 4. Benefits

손님 분포를 각 가게의 특성에 맞게 다르게 조절해서 개별화된 실험환경을 구축할 수 있다. 그로부터 현재 들어오는 손님분포에 변화를 주어, 보다 이상적인 손님 분포에 가까워지게끔 마케팅 전략을 구축할 수 있다. 또한 대략적인 계산을 통해 혼밥 손님을 받을지 안받을지를 정하기보다도 보상함수를 비롯한 강화학습의 결과를 바탕으로 데이터에 근거한 결정을 내릴 수 있을 것이다. 더나아가 어떤 배치가 가장 최적의 이득을 낼 수 있는지 알아봄으로써 가게 입장에서는 혼밥 손님을 놓치면서 얻지 못할 뻔 했던 잠재적 수익을 가져갈 수 있고, 혼밥하는 손님 입장에서도 더 많은 식당에서 밥을 먹을 수 있는 기회를 얻을 수 있다.

# 5. Member
- [Changhyeon Nam](https://github.com/changhyeonnam)
- [Yongkyun Lee](https://github.com/ComFromBehind)

# 6. Reference

- [Simple statistical gradient-following algorithms for connectionist reinforcement learning(REINFORCE)](https://link.springer.com/article/10.1007/BF00992696)
- [Resource Management with Deep Reinforcement Learning](https://people.csail.mit.edu/alizadeh/papers/deeprm-hotnets16.pdf)
- [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347.pdf)
