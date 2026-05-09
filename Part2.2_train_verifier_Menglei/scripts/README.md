## Example Output and Training Behavior


To demonstrate what the verifier is expected to learn, we use a simple math verification example. Given a question and a candidate solution, the verifier should not only decide whether the answer is correct, but also generate a short step-by-step verification rationale. 

the expected verifier output should explain that the solution correctly computes the total number of pies as 35, correctly subtracts the 8 dropped pies, and then ends with `Verification: Is the answer correct (Yes/No)? Yes`. 

For an incorrect candidate, such as a solution that calculates an editing percentage using `90 / 7.5 × 100` instead of converting 90 minutes into 1.5 hours first, the verifier should identify the earliest fatal error and output `Verification: Is the answer correct (Yes/No)? No`. 
### Example Input

```text
Question:
Marcus can fit 5 pies in his oven at once. He bakes 7 batches of pies, then drops 8 of them. How many pies are left?

Candidate Solution:
Marcus bakes 5 * 7 = 35 pies. After dropping 8 pies, he has 35 - 8 = 27 pies left. The answer is 27.


Step 1: The solution correctly computes the total number of pies baked as 5 * 7 = 35.

Step 2: The solution correctly subtracts the 8 dropped pies, giving 35 - 8 = 27.

Step 3: The final answer is consistent with the expected reasoning.

Verification: Is the answer correct (Yes/No)? Yes

Example Incorrect Candidate
Question:
Carol spends 4 hours writing a song, half that much time recording it, and 90 minutes editing it. What percentage of her total work time did she spend editing?

Candidate Solution:
Writing takes 4 hours. Recording takes 2 hours. Editing takes 90 minutes. Total time is 7.5 hours. The editing percentage is 90 / 7.5 * 100 = 12%. The answer is 12.


Step 1: The solution correctly computes the writing time as 4 hours and the recording time as 2 hours.

Step 2: The solution correctly states that 90 minutes is the editing time, but it fails to convert 90 minutes into hours before computing the percentage.

Step 3: The correct editing time is 1.5 hours. The percentage should be 1.5 / 7.5 * 100 = 20%, not 12%.

Earliest Error: The candidate uses 90 minutes directly in the percentage calculation instead of converting it to 1.5 hours.

Verification: Is the answer correct (Yes/No)? No


```


During training, we monitor the SFT loss curve to check whether the model is learning stably. 

A reasonable training curve does not need to decrease perfectly at every step. 
<img width="663" height="681" alt="image" src="https://github.com/user-attachments/assets/9039bf4b-dcbb-47aa-b78b-790d6f0a83cc" />

<img width="680" height="721" alt="image" src="https://github.com/user-attachments/assets/6f0bf683-68e9-44e5-b1c1-42af38a51752" />


Because the dataset is relatively small and the batch size may be limited by GPU memory, small local oscillations are normal. 

What we expect is a loss curve that fluctuates within a small range but shows an overall downward trend, which indicates that the model is gradually converging. 

When we include the training figures, the first figure can be used to show that the vanilla GenRM training is relatively stable: the loss has small short-term changes, but the global trend decreases, meaning the model is learning the verification format, rationale style, and final Yes/No prediction.

The second figure can be used to discuss the entropy-guided experiment. In our early test, the top-20% high-entropy token loss was not stably convergent before careful tuning. 

This is understandable because selecting only the highest-entropy 20% tokens is a relatively aggressive strategy, especially when the training set is not very large. Although high-entropy tokens often correspond to important reasoning forks, error transitions, or uncertain judgment points, the remaining lower-entropy tokens are still important for maintaining the output structure, such as `Step 1`, `Step 2`, `Verification: Is the answer correct (Yes/No)?`, and the final `Yes` or `No`. If these format-critical tokens receive too little training signal, the model may become unstable or produce malformed outputs.

Therefore, the hyperparameter design should be conservative and adjusted during SFT training.

We should continuously check whether the loss is decreasing, whether the output format remains valid, and whether the final verdict can still be parsed correctly. 


If the loss oscillates too heavily, diverges, or becomes unstable, we need to adjust the learning rate, entropy weight, top-k entropy ratio, batch size, gradient accumulation steps, maximum sequence length, or verdict-token weight. 

For this project, the pure top-20% entropy masking method is better treated as an ablation or preliminary experiment rather than the final stable method. 

A more reliable approach is verdict-aware entropy-weighted cross-entropy: all assistant tokens still receive loss, high-entropy reasoning tokens receive slightly larger weights, and final Yes/No verdict tokens are explicitly protected. This balances three goals at the same time: focusing more on high-value uncertain reasoning tokens, preserving stable convergence, and maintaining the required final judgment format. In summary, the training process should be monitored and adjusted dynamically. The ideal result is not an aggressively decreasing loss, but a stable loss curve with small oscillations and an overall convergence trend, together with verifier outputs that remain well-formatted and logically meaningful.
