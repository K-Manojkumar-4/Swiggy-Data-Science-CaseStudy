# Executive Summary — Swiggy Food Delivery Data Analysis (2025)

**Dataset**: 197,430 dish listings | 28 cities | Jan–Aug 2025  
**Analysis Date**: August 2025

---

## 1. Business Context

Swiggy operates in a highly competitive food-delivery market. Understanding pricing dynamics, customer rating behavior, and dish segmentation is critical for both the platform and partner restaurants. This analysis provides data-driven insights to support menu optimization, promotional strategy, and customer experience improvements.

---

## 2. Key Quantitative Findings

| Metric                          | Value                  |
|---------------------------------|------------------------|
| Total dish listings             | 197,430                |
| Average price                   | ₹268.51                |
| Median price                    | ₹229.00                |
| Price IQR                       | ₹139 – ₹329            |
| Average rating                  | 4.34 / 5.00            |
| Median rating                   | 4.40                   |
| Cities covered                  | 28                     |
| Zero / very low rating counts   | Significant share      |
| Price–Rating correlation        | Near zero (very weak)  |

---

## 3. Strategic Insights

1. **Geographic Concentration**  
   Bengaluru leads in listing volume. High-volume cities drive the majority of the catalog and should be prioritized for experiments and quality initiatives.

2. **Pricing Reality**  
   Most dishes sit in the affordable-to-mid range (₹100–₹350). Extreme high prices exist but are rare. Value perception is strong in the sub-₹200 segment.

3. **Rating vs Price Disconnect**  
   Higher price does not reliably translate into higher ratings. Quality perception is driven by factors beyond price (taste, packaging, consistency, delivery experience).

4. **Low Review Engagement**  
   A large fraction of dishes have zero or near-zero rating counts. This reduces ranking power, social proof, and discoverability.

5. **Segmentation Opportunity**  
   K-Means clustering reveals distinct groups of dishes that can be targeted differently (budget high-rated, premium, low-engagement, etc.).

6. **Limited Price Predictability from Ratings Alone**  
   Linear regression using only Rating + Rating Count yields R² ≈ 0.01. Price is influenced by many unobserved variables (cuisine, restaurant positioning, location tier, costs, competition).

---

## 4. Priority Recommendations

| Priority | Recommendation                                      | Expected Impact                          |
|----------|-----------------------------------------------------|------------------------------------------|
| High     | Incentivize ratings & reviews                       | Improve ranking, trust, conversion       |
| High     | Promote high-rated dishes under ₹200                | Increase order volume & value perception |
| Medium   | Expand mid/budget options in expensive cities       | Capture price-sensitive demand           |
| Medium   | Strengthen curation of “Recommended” section        | Higher conversion & satisfaction         |
| Medium   | Use cluster labels for targeted campaigns           | Better ROI on promotions                 |
| Long-term| Enrich data (cuisine, delivery SLA, discounts)      | Enable stronger predictive models        |

---

## 5. Conclusion

The analysis confirms that Swiggy’s catalog is rich and geographically diverse, yet customer engagement (ratings) and price–quality alignment remain areas for improvement. Simple unsupervised segmentation already provides actionable groups for marketing and menu decisions. Future work should focus on richer feature sets and more sophisticated models to support dynamic pricing and personalized recommendations.

---

*This summary is derived from the full end-to-end analysis notebook. Refer to the notebook and source modules for complete methodology, code, and visualizations.*
