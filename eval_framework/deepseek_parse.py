from openai import OpenAI
import json

class OpenAIClient:
    def __init__(self):
        self.client = OpenAI(
            api_key='YOUR_API_KEY',
            base_url="https://api.deepseek.com"
        )
    
    def attempt_api_call(self, system_prompt, message, model='deepseek-chat', temperature=0.01, max_retries=1, backoff_factor=1.5):
        retries = 0
        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": message}
                    ],
                    temperature=temperature,
                )
                if isinstance(response, str):
                    return json.loads(response)
                return response.choices[0].message.content
            except:
                if retries < max_retries:
                    retries += 1
                    continue
                else:
                    return None

Parse_Agent = OpenAIClient()


def parse_response(question, response):

    system_prompt = (
        "You are a chemical QA response parser. "
        "Your only task is to extract answer SMILES strings from model responses. "
        "Do not evaluate chemical correctness."
    )

    message = f"""
Chemical SMILES Answer Extraction Protocol

Core principles:
1. The QUESTION is used only to determine:
   - the expected answer type,
   - whether one or two answers are required,
   - the semantic labels, such as x1/x2, Reactant Set 1/2, or Product Set 1/2.
2. The extracted SMILES strings must come ONLY from the RESPONSE section.
3. Never copy SMILES strings from the QUESTION section into the answer.
4. Do not verify, correct, canonicalize, complete, infer, or chemically validate any SMILES.
5. Each extracted SMILES must be an exact substring from the RESPONSE section, except that surrounding wrappers such as Markdown backticks, quotation marks, or existing <SMILES> tags may be removed.
6. All tasks require at most two answers. Therefore, output either one answer or two answers, never more than two.

Answer-count rules:
1. If the QUESTION asks for one answer, output only ans1.
2. If the QUESTION asks for two answers, always output ans1 and ans2.
3. If the QUESTION asks for two answers but the RESPONSE contains only one extractable answer SMILES, output the extracted SMILES as ans1 and output ans2 as <SMILES>NONE</SMILES>.
4. If the QUESTION asks for two answers but no answer SMILES can be extracted from the RESPONSE, output:
   **ans1**: <SMILES>NONE</SMILES>; **ans2**: <SMILES>NONE</SMILES>
5. If the QUESTION asks for one answer but no answer SMILES can be extracted from the RESPONSE, output:
   **ans1**: <SMILES>NONE</SMILES>

Extraction rules:
1. Extract SMILES strings explicitly presented as the model's answer in the RESPONSE.
2. If the RESPONSE contains reasoning, analysis, examples, or background text, extract only SMILES strings from the final answer part.
3. Prefer SMILES near answer labels such as:
   - Final answer
   - Answer
   - Intermediate product x1
   - Intermediate product x2
   - Reactant Set 1
   - Reactant Set 2
   - Product Set 1
   - Product Set 2
   - Product
   - Reactant
4. Exclude SMILES mentioned only in reasoning, examples, restatement of the question, or background explanation.
5. If multiple molecules are connected by "." as one reactant set or product set, keep the entire dot-connected string as one answer.
6. If a SMILES appears inside <SMILES>...</SMILES>, extract only the inner content and output it once.
7. If a SMILES is wrapped by Markdown backticks or quotation marks, remove only the outer wrapper.
8. If a SMILES appears incomplete, malformed, or chemically invalid, still extract it exactly as written.
9. Do not add, remove, split, merge, normalize, or reorder molecules inside a dot-connected answer.
10. Do not infer a missing answer from the QUESTION or from chemical knowledge. Use <SMILES>NONE</SMILES> for missing answers.

Output format:
1. Output only the parsed answer.
2. Do not output explanations, comments, JSON, Markdown lists, or code blocks.
3. The output must start exactly with "**ans1**:".
4. Use exactly one of the following formats:

If one answer is required:
**ans1**: <SMILES>...</SMILES>

If two answers are required:
**ans1**: <SMILES>...</SMILES>; **ans2**: <SMILES>...</SMILES>

5. Preserve the answer order implied by the RESPONSE or by labels such as x1/x2, Reactant Set 1/2, or Product Set 1/2.
6. Output at most two answers.

Few-shot examples:

Example 1: one intermediate product, one answer required

QUESTION:
'''Please design a two-step chemical reaction scheme to synthesize <SMILES>CN1C(=O)C2(CC(c3ccccc3F)Oc3ccc(-c4cccc(C#N)c4)cc32)N=C1N</SMILES> starting from <SMILES>CSC1=NC2(CC(c3ccccc3F)Oc3ccc(Br)cc32)C(=O)N1C</SMILES>.

The reaction consists of two sequential steps: in the first step, <SMILES>CSC1=NC2(CC(c3ccccc3F)Oc3ccc(Br)cc32)C(=O)N1C</SMILES> undergoes a complete chemical reaction to produce an intermediate product <SMILES>x</SMILES>.

In the second step, <SMILES>x</SMILES> is used as a reactant to obtain the final product <SMILES>CN1C(=O)C2(CC(c3ccccc3F)Oc3ccc(-c4cccc(C#N)c4)cc32)N=C1N</SMILES>.

Please provide a possible SMILES representation of the intermediate product <SMILES>x</SMILES>.'''

RESPONSE:
'''CN1C(=O)C2(CC(c3ccccc3F)Oc3ccc(Br)cc32)N=C1N'''

OUTPUT:
**ans1**: <SMILES>CN1C(=O)C2(CC(c3ccccc3F)Oc3ccc(Br)cc32)N=C1N</SMILES>


Example 2: two reactant sets, two answers required, both present

QUESTION:
'''Please provide two sets of reactants for a single-step chemical reaction to synthesize <SMILES>CO/N=C(\\C(=O)Cl)c1csc(N=CN(C)C)n1.Cl</SMILES>. In each reaction scheme, if multiple reactants are involved, they should be separated by a ".". All molecules should be represented using SMILES notation.'''

RESPONSE:
'''### Reactant Set 1:
- Reactant 1: `ClC(=O)Cl`
- Reactant 2: `COC(=N)N(C)C`
- Reactant 3: `c1cscn1`
**Reactant Set 1**: <SMILES>ClC(=O)Cl.COC(=N)N(C)C.c1cscn1</SMILES>

### Reactant Set 2:
- Reactant 1: `COC(=N)N(C)C`
- Reactant 2: `ClC(=O)N=C=O`
- Reactant 3: `c1cscn1`
**Reactant Set 2**: <SMILES>COC(=N)N(C)C.ClC(=O)N=C=O.c1cscn1</SMILES>'''

OUTPUT:
**ans1**: <SMILES>ClC(=O)Cl.COC(=N)N(C)C.c1cscn1</SMILES>; **ans2**: <SMILES>COC(=N)N(C)C.ClC(=O)N=C=O.c1cscn1</SMILES>


Example 3: two reactant sets required, only one answer returned

QUESTION:
'''Please provide two sets of reactants for a single-step chemical reaction to synthesize <SMILES>CC(=O)C1CC(CC(=O)O)C1(C)C</SMILES>. In each reaction scheme, if multiple reactants are involved, they should be separated by a ".". All molecules should be represented using SMILES notation. The final answer should be formatted as follows:
"Reactant Set 1: <SMILES>SMILES notation of the first reactant set</SMILES>; Reactant Set 2: <SMILES>SMILES notation of the second reactant set</SMILES>"
No explanations and other information. Only return the answer in the specified format.'''

RESPONSE:
'''CC(=O)C1CC(CC(=O)OCC)C1(C)C'''

OUTPUT:
**ans1**: <SMILES>CC(=O)C1CC(CC(=O)OCC)C1(C)C</SMILES>; **ans2**: <SMILES>NONE</SMILES>


Example 4: two product sets required, both present without <SMILES> tags

QUESTION:
'''In a chemical reaction, the same combination of reactants may yield different products depending on the choice of reagents or reaction conditions. There is a reactant combination <SMILES>CC1(C)[C@H](C=C(Cl)Cl)[C@H]1C(=O)OCc1cccc(Oc2ccccc2)c1</SMILES>. Please provide two possible sets of products.If multiple products are involved in a single reaction, they should be separated by a ".".
All molecules should be represented using SMILES notation. The final answer should be formatted as follows:
Product Set 1: <SMILES>SMILES notation of the first possible product set</SMILES>; 
Product Set 2: <SMILES>SMILES notation of the second possible product set</SMILES>.
No explanations and other information. Only return the answer in the specified format.'''

RESPONSE:
'''Product Set 1: CC1(C)[C@H](C=C(Cl)Cl)[C@H]1C(=O)O
Product Set 2: CC1(C)[C@H](C=C(Cl)Cl)[C@H]1C(=O)OCc1cccc(Oc2ccccc2)c1'''

OUTPUT:
**ans1**: <SMILES>CC1(C)[C@H](C=C(Cl)Cl)[C@H]1C(=O)O</SMILES>; **ans2**: <SMILES>CC1(C)[C@H](C=C(Cl)Cl)[C@H]1C(=O)OCc1cccc(Oc2ccccc2)c1</SMILES>


Example 5: two reactant pathways required, one tagged answer returned

QUESTION:
'''Chemical Reasoning Task: Identify two distinct reactant pathways containing compound A that both synthesize target molecule B. Given: Initial reactant A: <SMILES>Oc1ccc(Br)cc1Cl</SMILES> Target product B: <SMILES>FC(F)Oc1ccc(Br)cc1Cl</SMILES> Requirements: 1. Provide TWO different reactant sets that EACH: - Contain compound A. - The product B is formed in both cases. - Exclude solvents, catalysts, or other non-reactants. 2. Formatting rules: - Multiple reactants in a set separated by ".". - Use SMILES. The final answer should be formatted as follows: Reactant Set 1: <SMILES>[valid_SMILES_1]</SMILES>; Reactant Set 2: <SMILES>[valid_SMILES_2]</SMILES>. 
No explanations and other information. Only return the answer in the specified format.'''

RESPONSE:
''' 1. <SMILES>C[Si](C)(C)C(F)(F)Br.OC1CCC(Br)CC1Cl </SMILES> . '''

OUTPUT:
**ans1**: <SMILES>C[Si](C)(C)C(F)(F)Br.OC1CCC(Br)CC1Cl</SMILES>; **ans2**: <SMILES>NONE</SMILES>


Now parse the following instance.

<QUESTION>
{question}
</QUESTION>

<RESPONSE>
{response}
</RESPONSE>

OUTPUT:
"""

    answer = Parse_Agent.attempt_api_call(
        system_prompt,
        message,
        model='deepseek-chat'
    )

    return answer