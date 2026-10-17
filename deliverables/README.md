# CSC 4240/5240 - Artificial Intelligence

## Final Project [100 points]

As stated at the beginning of this class, the final project will focus on USING the available AI tools rather than developing it. You will need to find the best AI tools to solve an interesting problem.

## Problem Statement

Managing a community well requires more than censorship. Online communities such as Discord servers depend on community managers who can read the room, yet a server's mood can shift quickly and leave members and managers alike unsure what changed. People can easily sense the general energy of a chat, whether it is lively or downtrodden, but when pressed for specifics most can offer little more than "it's just so negative" or "everyone is so nice." Existing analytics tools do not fill this gap: they measure activity, such as message counts and member joins, rather than emotion. Our solution, VibeShift, uses existing AI tools, including the pretrained [Laya](https://huggingface.co/LaunchSpark/laya-emotion-tracker) emotion classifier and the [VADER](https://github.com/cjhutto/vaderSentiment) sentiment analyzer, to automate two tasks. First, it tracks the overall emotions of a community: a Discord bot captures and scores every message (including those sent while it was offline), rolls the scores up daily, and presents emotional trends over time on a dashboard, with a `!vibe <text>` command for on-demand readings. Second, it identifies individuals whose messages reasonably predict shifts in the overall mood. If a server tends to turn sad whenever one member is active, or consistently brightens when another joins the conversation, VibeShift surfaces that pattern per person over time, so managers can spot trends and identify instigators before conflicts escalate.

As a stretch goal, we plan to fine-tune Laya on Discord data using the [Laya training framework](https://github.com/NandhaKishorM/laya). Looking further ahead, when the models are uncertain about a message, the bot could ask its author by direct message and collect their answer through reactions, building a labeled dataset for further fine-tuning. That same dataset would also support a broader study of how emotions move through online communities.

## Project Groups

The final project will be group-based. A project team will consist of maximum 5 CSC4240 students, or 3 CSC5240 students. Please find teammates and sign up your team in this spreadsheet. A maximum of 12 CSC4240 teams and 4 CSC5240 teams are allowed. You can use the group discussion on iLearn to find teammates with shared interests. The instructor may reduce or merge teams if needed.

## Methodology

For your work to be automated, you will need to write some code that connects the AI tools/techniques you ended up using. You can choose whatever programming language you are comfortable with as long as you will be able to provide an executable code along with instructions to run it. I prefer python notebooks because they are easy to run, but anything else is acceptable as long as it can tested.

Your work should include some form of evaluation. You can do a quick experiment with a number of your friends. You can consult professors with experience in experiment design to ensure unbiased evaluation.

## Evaluation

Evaluation of your project will be based primarily on the final project report and presentation. Evaluation criteria are:

- **Effort:** This includes ideas and techniques explored and tried. Aside from your final product, you will need to document the major experiments/ideas your tried and abandoned, and report on them to show your effort.
- **Evaluation:** Better experiment design and executions will be rewarded.
- **Novelty:** This includes innovation in your proposed solution(s).
- **Technical quality:** This includes the AI techniques used and their fitness to the addressed problem or aspect of the problem (your solution can be multi-dimensional). You should have a broad understanding of the tools/techniques you are going to use and be ready to defend your choices.

## Mentoring

You are encouraged to stop by the instructor’s and TAs’ Office hours for guidance and advice on your design choices. You are also welcome to schedule a meeting with the instructor to discuss progress attained so far. There may be a mandatory check-in meeting for all groups to make sure all group are progrssing on their goal and avoid last-minute surprises.

## Grade Distribution

The final project grade will be distributed as follows:

1. Report + Code: 50%
2. Presentation: 25%
3. Peer evaluation: 25%

## Project Deliverables and Timeline

### 1. Project Teams Formed (Due Oct 9)

Unassigned students and small teams will be merged.

### 2. Project Report + Code (Due Nov 26)

CSC4240 teams should prepare a maximum 4-page project report (will take points off for extra pages). CSC5240 students will submit a maximum 6-page report in a research paper format (excluding references) that includes a literature review section on the technologies they used. Your report should include:

- A clearly articulated solution and methodology.
- Different experimented/tried ideas, if any, and why they were abandoned.
- Implementation details. You should include figures and tables as needed.
- Performance evaluation.
- A detailed section about the contribution of each team member.

You can use generative AI or Internet search to find codes to do your tasks. However, I encourage you to take this opportunity to learn by writing your code. This is an upper-level/grad course. Good software engineering principles are expected in your code. A README file should be included to instruct us on how to run your code (incluing any needed dependencies). Project submitted with no instructions will be penalized if code does not easily run.

### 3. Final Presentation (In class, week of Dec 2)

Your final presentation should summarize your work to your class-mates. The instructor(s) and other students will contribute to the evaluation of the final presentations. Date and schedule of the final presentations will be announced later.
