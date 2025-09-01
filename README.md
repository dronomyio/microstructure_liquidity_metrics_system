# Intuition

The intuition behind these liquidity risk proxies relates to capturing different dimensions of how easily and cheaply you can trade without moving the market. Let me break down each measure:

## **Amihud Illiquidity Measure (|r|/Vol)**

The Amihud measure captures **price impact** - how much the price moves per dollar of trading volume. The intuition is straightforward:
- In liquid markets, you can trade large volumes without moving the price much (low |r|/Vol)
- In illiquid markets, even small trades cause significant price movements (high |r|/Vol)

Think of it like dropping a stone in water - in a deep pool (liquid market), the ripples are small; in a shallow puddle (illiquid market), the splash is dramatic. This measure is particularly useful because it only requires daily data (returns and volume) rather than intraday information.

## **Inter-trade Duration and Hazard Models**

These capture the **temporal dimension** of liquidity - how frequently trades occur:
- **Duration**: The time between consecutive trades. Longer durations suggest fewer market participants and lower liquidity
- **Hazard rate**: The instantaneous probability of a trade occurring, given no trade has happened yet

The intuition is that liquid assets trade continuously with short, predictable gaps between trades. Illiquid assets have long, unpredictable "dry spells." This matters because if you need to exit a position quickly, you want confidence that counterparties will appear. The hazard modeling helps capture how this trade arrival intensity varies with market conditions.

## **"Holes" in Time**

This extends the duration concept to identify **liquidity droughts** - abnormally long periods without trading:
- These gaps represent moments when the market effectively disappears
- They're particularly dangerous because they often coincide with stress periods when liquidity is most needed

The intuition is that liquidity risk isn't just about average conditions but about the worst-case scenarios. A market might seem liquid on average but have periodic "blackouts" where trading becomes impossible.

## **Why These Work as Spread/Depth Stand-ins**

These measures serve as good proxies for bid-ask spreads and market depth because they capture the same underlying friction:

1. **Price impact (Amihud) ≈ Spread + Depth effects**: When spreads are wide or depth is thin, trades move prices more
2. **Trade frequency ≈ Market presence**: Narrow spreads and deep books attract more trading; sparse trading suggests wide spreads/thin books
3. **Time gaps ≈ Market maker confidence**: Market makers pull back (widening spreads, reducing depth) during uncertain times, creating trading gaps

The key advantage is that these proxies can be computed from readily available trade data, while actual spread and depth data might be unavailable (especially historically) or expensive to obtain. They're particularly valuable for comparing liquidity across different markets or time periods where market microstructure differs.


# Math

Let me walk through the mathematical formulations of these liquidity risk proxies:

## **Amihud Illiquidity Measure**

The Amihud (2002) measure is defined as:

$$ILLIQ_{i,t} = \frac{1}{D_t} \sum_{d=1}^{D_t} \frac{|r_{i,d,t}|}{V_{i,d,t}}$$

Where:
- $|r_{i,d,t}|$ = absolute return of asset $i$ on day $d$ in period $t$
- $V_{i,d,t}$ = dollar trading volume on that day
- $D_t$ = number of trading days in period $t$

**Interpretation**: Average daily price impact per dollar traded. Units are typically $10^{-6}$ (price change per million dollars).

For high-frequency versions:
$$ILLIQ_{intraday} = \frac{|r_{t,t+\Delta t}|}{V_{t,t+\Delta t}}$$

This captures the price movement per unit volume over interval $\Delta t$.

## **Inter-trade Duration Models**

### Basic Duration
Let $t_i$ be the time of trade $i$. The duration is:
$$d_i = t_i - t_{i-1}$$

### Autoregressive Conditional Duration (ACD) Model
Engle & Russell (1998) model durations as:
$$d_i = \psi_i \cdot \epsilon_i$$

Where $\psi_i$ is the conditional expected duration:
$$\psi_i = \omega + \sum_{j=1}^{p} \alpha_j d_{i-j} + \sum_{j=1}^{q} \beta_j \psi_{i-j}$$

The error term $\epsilon_i$ follows a distribution (often exponential or Weibull) with unit mean.

### Hazard Function
The hazard rate (instantaneous probability of trade arrival) is:
$$h(t) = \frac{f(t)}{S(t)} = \frac{f(t)}{1-F(t)}$$

Where:
- $f(t)$ = probability density of duration
- $F(t)$ = cumulative distribution function
- $S(t)$ = survival function (probability no trade by time $t$)

For exponential durations with parameter $\lambda$:
$$h(t) = \lambda$$ (constant hazard)

For Weibull with scale $\lambda$ and shape $p$:
$$h(t) = \frac{p}{\lambda}\left(\frac{t}{\lambda}\right)^{p-1}$$

- $p > 1$: increasing hazard (trade more likely as time passes)
- $p < 1$: decreasing hazard (trade less likely as time passes)

## **"Holes" in Time Analysis**

### Identifying Holes
Define a "hole" as a duration exceeding threshold $\tau$:
$$H_i = \mathbb{1}[d_i > \tau]$$

Where $\tau$ might be:
- A multiple of median duration: $\tau = k \cdot \text{median}(d)$, typically $k \in [3,10]$
- A percentile: $\tau = \text{percentile}_{95}(d)$

### Hole Statistics
**Hole frequency**:
$$HF = \frac{\sum_{i=1}^{N} H_i}{N}$$

**Hole intensity** (proportion of time in holes):
$$HI = \frac{\sum_{i: H_i=1} d_i}{\sum_{i=1}^{N} d_i}$$

**Maximum hole duration**:
$$MH = \max_{i: H_i=1} d_i$$

## **Relationship to Spread and Depth**

### Kyle's Lambda (Price Impact)
In Kyle (1985), permanent price impact is:
$$\Delta p = \lambda \cdot v$$

Where $v$ is net order flow. The Amihud measure approximates this:
$$\lambda \approx \frac{|r|}{V} \approx \frac{s}{2D} + \frac{1}{2\kappa}$$

Where:
- $s$ = bid-ask spread
- $D$ = market depth at best quotes
- $\kappa$ = market depth beyond best quotes

### Roll's Implied Spread
From serial covariance of returns:
$$\text{Cov}(r_t, r_{t-1}) = -\frac{s^2}{4}$$

Therefore:
$$s = 2\sqrt{-\text{Cov}(r_t, r_{t-1})}$$

### Hasbrouck's Information Share
Decomposing price variance into information and noise:
$$\text{Var}(p_t) = \sigma^2_m + \sigma^2_s$$

Where trades with higher durations typically have higher information content.

## **Composite Liquidity Score**

Often these are combined into a single measure:
$$L_{composite} = w_1 \cdot \text{rank}(ILLIQ) + w_2 \cdot \text{rank}(\bar{d}) + w_3 \cdot \text{rank}(HI)$$

Or using principal components:
$$L_{PC1} = a_1 \cdot \log(ILLIQ) + a_2 \cdot \log(\bar{d}) + a_3 \cdot \log(HI)$$

Where $a_i$ are the first principal component loadings.

## **Statistical Properties**

These measures typically exhibit:
1. **Heavy tails**: Log-normal or power-law distributions
2. **Clustering**: High autocorrelation (liquidity droughts cluster)
3. **Common factors**: Strong cross-sectional correlation during crises

The math reveals that these proxies capture complementary aspects of liquidity - price impact (Amihud), market presence (duration), and tail risk (holes) - making them powerful tools for risk management when direct microstructure data is unavailable.

# Relevant Papers

I should clarify that I was synthesizing from multiple foundational papers in market microstructure, not following a single source. Let me provide the specific paper references for each component:

## **Primary Papers for Each Measure**

### Amihud Illiquidity Measure
- **Amihud, Y. (2002)**. "Illiquidity and stock returns: cross-section and time-series effects." *Journal of Financial Markets*, 5(1), 31-56.
  - This is the original paper introducing the |r|/Volume measure

### Inter-trade Duration Models
- **Engle, R. F., & Russell, J. R. (1998)**. "Autoregressive conditional duration: a new model for irregularly spaced transaction data." *Econometrica*, 66(5), 1127-1162.
  - Introduced the ACD model I described

- **Bauwens, L., & Giot, P. (2000)**. "The logarithmic ACD model: an application to the bid-ask quote process of three NYSE stocks." *Annales d'Economie et de Statistique*, 117-149.
  - Extended ACD models with hazard function analysis

### Price Impact and Market Microstructure Theory
- **Kyle, A. S. (1985)**. "Continuous auctions and insider trading." *Econometrica*, 53(6), 1315-1335.
  - The lambda (λ) price impact model

- **Roll, R. (1984)**. "A simple implicit measure of the effective bid-ask spread in an efficient market." *Journal of Finance*, 39(4), 1127-1139.
  - The Roll implied spread measure from serial covariance

### Hazard Models in Finance
- **Dufour, A., & Engle, R. F. (2000)**. "Time and the price impact of a trade." *Journal of Finance*, 55(6), 2467-2498.
  - Connected duration to price impact

### "Holes" in Trading
The concept of "holes" or gaps in trading doesn't come from a single seminal paper but appears in various forms in:
- **Easley, D., & O'Hara, M. (1992)**. "Time and the process of security price adjustment." *Journal of Finance*, 47(2), 577-605.
- Literature on "no-trade" periods and information models

## **Comprehensive Surveys**

If looking for a unified treatment, these surveys cover multiple measures:

- **Goyenko, R. Y., Holden, C. W., & Trzcinka, C. A. (2009)**. "Do liquidity measures measure liquidity?" *Journal of Financial Economics*, 92(2), 153-181.
  - Compares various liquidity proxies including Amihud

- **Hasbrouck, J. (2009)**. "Trading costs and returns for US equities: Estimating effective costs from daily data." *Journal of Finance*, 64(3), 1445-1477.
  - Discusses various trade-based proxies

The mathematical framework presented synthesizes across these papers rather than coming from a single source. Is there a particular aspect or specific paper you'd like me to elaborate on?


