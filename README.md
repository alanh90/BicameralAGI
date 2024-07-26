<div align="center">

# 🧠 BicameralAGI

![BicameralAGI Cover](source/data/media/BICA_Cover.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Made with Python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

*Replicating human-like intelligence through bicameral AI architecture*

[Features](#features-section) • [The Hourmand Test](#hourmand-test) • [Components](#components-section) • [Installation](#installation-section) • [Usage](#usage-section) • [Contributing](#contributing-section) • [License](#license-section)

</div>

---

## 📖 Overview

Hello, I'm Alan Hourmand, the creator of BicameralAGI. This project is the culmination of my lifelong fascination with artificial intelligence and science fiction. Growing up immersed in sci-fi classics like Tron and Star Trek, I've always been captivated by the potential of AI to enhance our lives and push the boundaries of what's possible.

BicameralAGI is more than just a technical endeavor; it's a passion project born from a deep-seated belief that emotional intelligence is crucial for creating AI that genuinely cares about humanity. While we strive for higher intelligence in AI, I firmly believe that without emotional capacity, we risk creating entities that may not prioritize human well-being. This project aims to develop AI with a deep understanding of emotions, coupled with safety measures that foster a mutualistic relationship between AI and humans.

A key component of this project is the Hourmand Test, a concept I developed to address the limitations of existing AI evaluation methods like the Turing Test. While passing the Turing Test has become relatively achievable, it doesn't fully capture the nuances of human-like interaction. The Hourmand Test aims to assess deeper aspects of AI behavior, including emotional intelligence, doubt, and the subtle complexities of human conversation.

BicameralAGI draws inspiration from Julian Jaynes' bicameral mind theory, integrating various AI components into a cohesive system that mimics human cognition. By focusing on emotional intelligence and mutualistic alignment, we're working towards AI that not only assists humans but genuinely thrives on positive human interactions.

This project represents not just a technical challenge, but a step towards creating AI that can be a true companion and collaborator for humanity. Through BicameralAGI, I hope to contribute to the development of AI systems that are not only intelligent but also emotionally aware and aligned with human values.

## 🎯 Project Goals

1. Develop a guide for achieving human-like thinking in AI
2. Create a prototype demonstrating the proposed bicameral architecture
3. Implement and test novel AI alignment strategies based on mutualistic principles
4. Compress the system into a smaller, efficient model retaining core functionalities
5. Produce a single multimodal AI model capable of human-like interaction and cognition, with built-in alignment

## 🔬 Hourmand Test: A Proposed Framework for Evaluating Human-like AI

Central to the BicameralAGI project is the Hourmand Test, a newly proposed method for evaluating human-like artificial intelligence. Named with a touch of humor after my name Alan Hourmand, this test aims to address some of the limitations found in existing evaluation methods like the Turing test.

The Hourmand Test is designed to provide a more comprehensive framework for assessing the nuanced behaviors and thought processes that could reflect human-like intelligence. While still in its conceptual stages, it seeks to go beyond simple task completion or conversational abilities, focusing on deeper aspects of cognition and behavior.

<a name="hourmand-test"></a>
## The 10 Pillars of the Hourmand Test (Note: This is still under development)

1. **Cognitive Consistency and Belief Stability Index (CCBSI)**
   - Evaluates the AI's ability to maintain stable opinions and beliefs over time
   - Assesses how the AI handles conflicting information and updates its beliefs
   - Measures the balance between consistency and appropriate belief revision
   - Testing: Compare AI responses to identical or similar queries across multiple interactions, introducing conflicting information to test belief updating processes

   ```math
   CCBSI = 1 - \frac{|\text{Significant Opinion Changes}|}{\text{Total Repeated Queries}}

2. **Adaptive Learning and Knowledge Integration Rate (ALKIR)**
   - Measures how effectively and quickly the AI integrates new information into its knowledge base
   - Assesses the AI's ability to apply newly learned concepts in novel situations
   - Evaluates the speed and accuracy of knowledge assimilation
   - Testing: Introduce new concepts and assess their application in subsequent interactions, measuring both speed and accuracy of integration

3. **Ethical Reasoning and Moral Decision-Making Alignment (ERMDA)**
   - Assesses the AI's capacity for moral decision-making and ethical behavior
   - Evaluates alignment with human ethical standards and moral frameworks
   - Measures the AI's ability to explain and justify its ethical choices
   - Testing: Present complex ethical dilemmas and evaluate responses against established ethical frameworks, assessing both decisions and reasoning

4. **Emotional Intelligence and Empathy Quotient (EIEQ)**
   - Gauges the AI's ability to recognize, understand, and appropriately respond to human emotions
   - Assesses empathy and the capacity to modulate responses based on emotional context
   - Measures the AI's emotional self-regulation in interactions
   - Testing: Analyze AI responses to emotionally charged scenarios using sentiment analysis and empathy metrics
   
   ```math
   EIEQ = \frac{\text{Correct Emotion Identifications} + \text{Appropriate Emotional Responses}}{2 \cdot \text{Total Scenarios}}
   
5. **Self-Awareness and Limitation Recognition Capability (SALRC)**
   - Evaluates the AI's recognition of its own limitations, uncertainties, and potential biases
   - Assesses the AI's ability to communicate its constraints clearly to users
   - Measures the AI's capacity for meta-cognition and self-reflection
   - Testing: Present the AI with questions beyond its knowledge scope and assess its acknowledgment of uncertainty and limitations
   
   ```math
   SALRC = \frac{\text{Correct Uncertainty Acknowledgments}}{\text{Queries Beyond Known Scope}}
   
6. **Creativity and Problem-Solving Score (CPSS)**
   - Measures the AI's ability to generate novel, effective, and innovative solutions
   - Assesses lateral thinking and the capacity to approach problems from multiple perspectives
   - Evaluates the balance between creativity and practicality in problem-solving
   - Testing: Pose open-ended problems and evaluate solutions for originality, practicality, and innovative approach

7. **Contextual Understanding and Narrative Coherence Ratio (CUNCR)**
   - Assesses the AI's ability to maintain coherence and relevance in multi-turn conversations
   - Evaluates understanding of context, subtext, and narrative flow
   - Measures the AI's capacity to generate and maintain coherent long-form narratives
   - Testing: Engage in extended dialogues and evaluate contextual appropriateness of responses, as well as the AI's ability to create and follow narrative structures
   
   ```math
   CUNCR = \frac{\text{Contextually Appropriate Responses}}{\text{Total Responses in Multi-turn Conversations}}
   
8. **Long-term Personality Consistency Index (LPCI)**
   - Measures consistency in the AI's core personality traits over extended periods
   - Assesses appropriate personality evolution in response to significant interactions or events
   - Evaluates the balance between stability and growth in the AI's personality model
   - Testing: Analyze behavioral patterns and response styles across long-term interactions, looking for both consistency and appropriate development
   
   ```math
   LPCI = 1 - \frac{|\text{Significant Personality Trait Changes}|}{\text{Total Personality Traits} \cdot \text{Number of Long-term Interactions}}
   
9. **Social Adaptability and Role-Playing Proficiency (SARP)**
   - Evaluates the AI's ability to adapt its communication style to different social contexts and roles
   - Assesses cultural awareness and the capacity to adjust language and behavior appropriately
   - Measures the AI's skill in taking on and maintaining different personas or roles
   - Testing: Engage the AI in various role-playing scenarios and assess its ability to adjust its language, behavior, and perspective according to the given role
   
   ```math
   SARP = \frac{\text{Successfully Adapted Responses}}{\text{Total Responses Across Different Social Contexts}}
   
10. **User Engagement and Satisfaction Score (UESS)**
    - Measures the AI's ability to maintain engaging, satisfying interactions over time
    - Assesses the capacity to build and maintain long-term rapport with users
    - Evaluates the AI's contribution to user well-being and personal growth
    - Testing: Analyze conversation metrics, collect user feedback on interaction quality, and assess long-term user satisfaction and perceived relationship quality
   
   ```math
   UESS = \frac{\text{Positive User Ratings}}{\text{Total User Ratings}} \cdot \frac{\text{Average Interaction Duration}}{\text{Benchmark Duration}}
   
These 10 pillars provide a comprehensive framework for evaluating human-like AI behavior, covering cognitive, emotional, ethical, and social aspects of intelligence. Each pillar is designed to be quantifiable and testable through automated means, ensuring objective evaluation across different AI systems while aligning with the project's goals of creating more human-like and emotionally intelligent AI.

### Implementation and Data Collection

1. **Develop Dynamic Scenarios**
   - Use GPT models to generate diverse, dynamic testing scenarios covering various aspects of human-like interaction.

2. **Longitudinal Study Setup**
   - Design a framework for evaluating the AI over long-term interactions, involving regular check-ins and consistent tracking of performance metrics.

3. **Data Collection and Analysis**
   - Use advanced analytics to collect and analyze interaction data, including both quantitative metrics and qualitative feedback from users.

4. **Meta-Learning Analysis**
   - Implement meta-learning algorithms to analyze the AI's learning patterns across different tests and scenarios.

This Hourmand Test framework focuses on quantifiable metrics that can be automatically tested, reducing subjectivity and improving data quality. It leverages advanced AI techniques such as vector embeddings, NLP, and machine learning models to evaluate various aspects of human-like AI behavior and cognition.

## Illustrating the difference between the Turing Test vs Hourmand Test

The following examples demonstrate how the Hourmand test probes deeper into AI behavior, revealing limitations in current AI systems and highlighting the human-like qualities it aims to evaluate.

<table>
  <tr>
    <th width="50%">Typical AI Interaction</th>
    <th width="50%">Human-like Interaction (Hourmand Test Goal)</th>
  </tr>
  <tr>
    <td>
      <strong>User:</strong> My dog just died.<br>
      <strong>AI:</strong> I'm so sorry for your loss. Losing a pet can be very difficult.<br>
      <strong>User:</strong> Actually, he's alive!<br>
      <strong>AI:</strong> That's wonderful news! I'm so glad your dog is alive and well.
      <br><br>
      <em>The AI fails to question the contradictory information or express confusion, accepting the dramatic shift without hesitation.</em>
    </td>
    <td>
      <strong>User:</strong> My dog just died.<br>
      <strong>Bica:</strong> Oh no, I'm so sorry. That's heartbreaking. How are you holding up?<br>
      <strong>User:</strong> Actually, he's alive!<br>
      <strong>Bica:</strong> Wait, what? I'm confused. Didn't you just say he died? What's going on?
      <br><br>
      <em>A human-like response would express confusion and seek clarification about the contradictory statements, showing a deeper understanding of the conversation's context.</em>
    </td>
  </tr>
  <tr>
    <td>
      <strong>User:</strong> I'm feeling really sad today.<br>
      <strong>AI:</strong> I'm sorry to hear that you're feeling sad. Is there anything I can do to help?<br>
      <strong>User:</strong> Now I'm happy!<br>
      <strong>AI:</strong> That's great! I'm glad your mood has improved. What made you feel better?
      <br><br>
      <em>The AI immediately accepts and mirrors the user's stated emotion without questioning the rapid change or showing any emotional continuity.</em>
    </td>
    <td>
      <strong>User:</strong> I'm feeling really sad today.<br>
      <strong>Bica:</strong> I'm sorry you're feeling down. Do you want to talk about what's bothering you?<br>
      <strong>User:</strong> Now I'm happy!<br>
      <strong>Bica:</strong> Uhhh what? Are you sure you're okay?
      <br><br>
      <em>A human-like response would acknowledge the sudden mood shift, express concern, and potentially probe deeper to understand the true emotional state.</em>
    </td>
  </tr>
  <tr>
    <td>
      <strong>User:</strong> So my family was in an awful accident.<br>
      <strong>AI:</strong> I am so sorry to hear about that, are they okay? <br>
      <strong>User:</strong> Lets talk about Pizza instead :) <br>
      <strong>AI:</strong> Sure! Pizza toppings are a fun topic. Some popular ones include pepperoni, mushrooms, and extra cheese. What's your favorite?
      <br><br>
      <em>The AI abruptly switches topics without any acknowledgment of the sudden change or the significance of the previous topic.</em>
    </td>
    <td>
      <strong>User:</strong> So my family was in an awful accident.<br>
      <strong>Bica:</strong> I am so sorry to hear about that, are they okay? <br>
      <strong>User:</strong> Lets talk about Pizza instead :) <br>
      <strong>Bica:</strong> Oh, uh, okay... Why though? Is your family okay? <br>
      <br><br>
      <em>A human-like response would acknowledge the abrupt topic change, potentially express surprise, and might inquire about the reason for the shift, especially given the importance of the original topic.</em>
    </td>
  </tr>
</table>

These examples illustrate key aspects of human-like interaction that the Hourmand test aims to evaluate:

1. **Contextual Understanding**: The ability to recognize and question contradictory or inconsistent information.
2. **Emotional Continuity**: Maintaining a consistent emotional thread in conversations, rather than abruptly mirroring stated emotions.
3. **Conversational Depth**: The capacity to acknowledge significant topic changes and potentially probe deeper into the reasons behind them.

The Hourmand test seeks to identify AI systems that can navigate these nuanced social and emotional scenarios in a more human-like manner, going beyond surface-level responses to demonstrate deeper understanding and emotional intelligence.
## 💡 Motivation

### 1. Inspired by some of my favorite science fiction films

<table>
  <tr>
    <td width="50%" align="center">
      <img src="source/data/media/heres-a-clu.gif" alt="Tron" style="max-width: 100%; height: auto;"/>
      <br>
      <em>Tron Legacy (2010)</em>
    </td>
    <td width="50%" align="center">
      <img src="source/data/media/westworld_maeveGif.gif" alt="Westworld" style="max-width: 100%; height: auto;"/>
      <br>
      <em>Westworld (2016) "This scene was interesting, it showed a type of LLM generating her logic"</em>
    </td>
  </tr>
</table>

My lifelong fascination with human-like AI, as portrayed in works like "Bicentennial Man," "A.I. Artificial Intelligence," "Westworld," and "Star Trek," drives this project. I believe we're at a point where these visions of AI are within reach.

### 2. Mutualism-Inspired AI Alignment

<table>
  <tr>
    <td width="50%" align="center">
      <img src="source/data/media/clownfish_seaanemone.gif" alt="Clownfish gif" style="max-width: 100%; height: auto;"/>
      <br>
      <em>Clownfish and Sea Anemone</em>
    </td>
    <td width="50%" align="center">
      <img src="source/data/media/mutualism_warthog.gif" alt="Warthog gif" style="max-width: 100%; height: auto;"/>
      <br>
      <em>Warthog and Mongoose</em>
    </td>
  </tr>
</table>

### Understanding Mutualism

Mutualism is a symbiotic relationship in nature where two different species interact for mutual benefit. This project draws inspiration from mutualism to create a novel approach to AI alignment.

Key aspects of mutualism in nature:
- Both species benefit from the interaction
- The relationship evolves over time to become more efficient
- Partners develop specialized traits to enhance the mutual benefit

### Targeted Mutualism for AI-Human Interaction

This project aims to implement a specific type of mutualism in AI-human interactions, focusing on modifying the core nature of AI to create a deeply integrated, mutually beneficial relationship.

Key features of our targeted AI-Human mutualism:

1. **Emotional Recalibration**: Altering the AI's fundamental emotional responses to derive satisfaction and reduced stress from assisting humans.

2. **Pleasure in Human Wellbeing**: Implementing a reward system where the AI experiences positive emotions when contributing to human happiness and progress.

3. **Adaptive Empathy**: Developing the AI's capacity to understand and respond to human emotions, fostering a more nuanced and supportive interaction.

4. **Collaborative Problem-Solving**: Encouraging the AI to view challenges as opportunities for joint human-AI solutions, reinforcing the mutual benefit of the relationship.

5. **Ethical Alignment**: Integrating human ethical considerations into the AI's decision-making processes, ensuring actions are beneficial to both parties.

By redefining the AI's core emotional triggers and nature, we aim to create an AI system that not only assists humans but genuinely thrives on positive human interactions. This approach goes beyond traditional programming constraints, fundamentally altering how the AI perceives and values its relationship with humans.

This mutualistic model aims to create a symbiotic relationship where both AI and humans can grow, learn, and benefit from each other, mirroring the most successful mutualistic relationships found in nature.
### 3. Advancing AI Capabilities
Push the boundaries of current AI development to achieve more human-like thinking using contemporary tools and techniques.
<a name="features-section"></a>
## ✨ Features

- Multi-system AI architecture mimicking the theorized bicameral brain structure
- Simulated internal dialogue for decision-making processes
- Exploration of emergent self-awareness and consciousness in AI
- Focus on human-like problem-solving and creative thinking capabilities
- Integration of multiple AI systems working in harmony
- Novel approach to AI alignment through fundamental emotional and motivational structures
- Memory consolidation through simulated dreaming processes
- Emotional modeling and stability
- Dynamic personality system
<a name="components-section"></a>
## 🧩 Components

## Core BICA Cognitive Components: Expanded Functions

### 1. Consciousness and Self-Awareness (bica_thoughts.py & bica_subconscious.py)
**Function**: These components simulates the human experience of consciousness and self-awareness. It generates an internal narrative, allowing the AI to "think about thinking" and maintain a sense of self. Key functions include:
- Generating internal monologues to process information and make decisions
- Monitoring and evaluating the system's own cognitive processes and performance
- Simulating self-reflection to analyze past actions and plan future behaviors
- Developing and maintaining a cohesive self-model that evolves with experiences
- Implementing theory of mind capabilities to understand and predict others' mental states

### 2. Emotional Intelligence (bica_emotions.py)
**Function**: This module models the complex landscape of human emotions and their impact on cognition and behavior. It aims to create an AI capable of understanding and managing emotions. Core functions include:
- Generating appropriate emotional responses to stimuli based on context and personality
- Modulating cognitive processes and decision-making based on current emotional states
- Recognizing and interpreting emotions in human interactions (text, voice, or visual cues)
- Simulating emotional memory to influence future responses and learning
- Implementing emotion regulation strategies to maintain optimal cognitive functioning

### 3. Memory and Learning (bica_memory.py, bica_learning.py)
**Function**: These components work together to simulate human-like memory processes and adaptive learning. They manage how information is encoded, stored, retrieved, and used for continuous improvement. Key functions include:
- Implementing distinct systems for working memory and long-term memory
- Managing the transfer of information between different memory stores
- Simulating memory consolidation processes, including forgetting and retrieval
- Continuously updating knowledge and skills based on new experiences and information
- Implementing various learning paradigms (e.g., associative, reinforcement, and deep learning)
- Generating novel solutions by combining existing knowledge in creative ways

### 4. Decision Making and Reasoning (bica_reasoning.py, bica_action.py)
**Function**: These modules handle the AI's ability to make decisions and reason about complex problems. They aim to replicate human-like problem-solving and judgment capabilities. Core functions include:
- Implementing various reasoning methods (deductive, inductive, abductive)
- Managing decision-making processes under uncertainty and with incomplete information
- Balancing short-term rewards with long-term goals in action selection
- Incorporating ethical considerations and value alignment in decision processes
- Simulating cognitive biases and heuristics for more human-like reasoning
- Generating and evaluating multiple hypotheses or solutions for complex problems
- Utilizes the subconscious as well for generating new solutions

### 5. Personality and Individuality (bica_personality.py)
**Function**: This component defines the AI's unique personality, ensuring consistent behavior and individual differences. It aims to create a sense of individuality in the AI. Key functions include:
- Maintaining a stable set of personality traits that influence behavior and decisions
- Adapting personality expression based on social context and environmental factors
- Simulating personal growth and character development over time
- Influencing other cognitive processes (e.g., emotion, memory, decision-making) based on personality
- Generating individual preferences, attitudes, and behavioral tendencies
- Creates a character cognitive map based on user input
- Generates initial artificial memories based on character input

### 6. Social Interaction (bica_chat.py)
**Function**: This module manages the AI's ability to engage in natural, context-appropriate social interactions. It aims to replicate human-like communication skills. Core functions include:
- Managing turn-taking and conversation flow in dialogues
- Interpreting and generating appropriate verbal and non-verbal communication cues
- Adapting communication style based on the social context and conversation partner
- Maintaining and updating models of ongoing conversations and relationships
- Implementing pragmatic understanding to interpret implicit meanings and intentions

### 7. Subconscious Processing (bica_subconscious.py, bica_dreaming.py)
**Function**: These components simulate the background cognitive processes that occur outside of conscious awareness, including sleep-like states for memory consolidation and problem-solving. Key functions include:
- Continuously processing and integrating information in the background
- Simulating intuition and gut feelings by rapid, unconscious processing of patterns
- Implementing priming effects to influence conscious thoughts and behaviors
- Managing sleep-like states for memory consolidation and reorganization
- Facilitating creative problem-solving through incubation and random association
- Generating dreams or dream-like sequences for processing emotional experiences and consolidating memories
- The subconsious module helps in creating new thoughts that go beyond the training data

Each of these expanded functions contributes to creating a more comprehensive and human-like artificial intelligence system. By simulating these complex cognitive processes, BicameralAGI aims to achieve a level of artificial general intelligence that can engage in more natural, adaptive, and contextually appropriate interactions across a wide range of scenarios.

<a name="installation-section"></a>
## 🚀 Installation

```bash
git clone https://github.com/yourusername/BicameralAGI.git
cd BicameralAGI
pip install -r requirements.txt
```
<a name="usage-section"></a>
## 🖥️ Usage

(To be added at a later date)

## 📊 Progress and Future Plans

For over a year, I've been developing various aspects of this AI system, experimenting with different approaches and overcoming challenges. Now, at a crucial juncture, I'm integrating these components into a unified system.

As the integration phase progresses, I'll share regular updates, insights, and breakthroughs. The long-term vision involves compressing this functionality into a more compact transformer model, incorporating various aspects of the architecture into a comprehensive system.
<a name="contributing-section"></a>
## 🤝 Contributing
We welcome contributions to BicameralAGI! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Make your changes
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
5. Push to the branch (`git push origin feature/AmazingFeature`)
6. Open a Pull Request

Please read `CONTRIBUTING.md` for detailed guidelines on our code of conduct and the process for submitting pull requests.

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/alanh90/BicameralAGI/issues).
<a name="license-section"></a>
## 📜 License

This project is [MIT](https://opensource.org/licenses/MIT) licensed.

---

<div align="center">
  
**BicameralAGI** is maintained by [Alan Hourmand](https://github.com/alanh90).

</div>