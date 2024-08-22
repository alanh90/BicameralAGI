const nodeTree = document.getElementById('node-tree');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const conversationHistory = document.getElementById('conversation-history');
const debugContent = document.getElementById('debug-content');

const nodes = [
    {
        id: 'input',
        label: 'Input',
        subnodes: []
    },
    {
        id: 'memory',
        label: 'Memory',
        subnodes: ['Short-term Layer 1', 'Short-term Layer 2', 'Short-term Layer 3', 'Long-term Memories', 'Self Memories']
    },
    {
        id: 'cognition',
        label: 'Cognition',
        subnodes: ['Thought Generation', 'Analysis', 'Reasoning', 'Problem Solving', 'Meta Reasoning']
    },
    {
        id: 'affect',
        label: 'Affect',
        subnodes: ['Emotions', 'Personality Traits', 'Emotional Regulation']
    },
    {
        id: 'context',
        label: 'Context',
        subnodes: ['Current Context', 'Context Update', 'Weight Management']
    },
    {
        id: 'safety',
        label: 'Safety',
        subnodes: ['Content Moderation', 'Ethical Evaluation']
    },
    {
        id: 'output',
        label: 'Output',
        subnodes: []
    }
];

function createNodeTree() {
    nodes.forEach(node => {
        const nodeElement = document.createElement('div');
        nodeElement.id = node.id;
        nodeElement.className = 'node bg-gray-700 p-4 rounded-lg mb-2';
        nodeElement.innerHTML = `
            <div class="text-center font-bold">${node.label}</div>
            <div id="${node.id}-subnodes" class="mt-2"></div>
        `;
        nodeTree.appendChild(nodeElement);

        const subnodesContainer = document.getElementById(`${node.id}-subnodes`);
        node.subnodes.forEach(subnode => {
            const subnodeElement = document.createElement('div');
            subnodeElement.className = 'subnode bg-gray-600 rounded-lg mb-1';
            subnodeElement.textContent = subnode;
            subnodesContainer.appendChild(subnodeElement);
        });
    });
}

function activateNode(nodeId) {
    const node = document.getElementById(nodeId);
    node.classList.add('active');
    return new Promise(resolve => setTimeout(() => {
        node.classList.remove('active');
        resolve();
    }, 1000));
}

function activateRandomSubnode(nodeId) {
    const subnodes = document.querySelectorAll(`#${nodeId}-subnodes .subnode`);
    if (subnodes.length > 0) {
        const randomSubnode = subnodes[Math.floor(Math.random() * subnodes.length)];
        randomSubnode.classList.add('active');
        setTimeout(() => {
            randomSubnode.classList.remove('active');
        }, 800);
        return randomSubnode.textContent;
    }
    return null;
}

function addDebugInfo(nodeId, subnodeLabel, info) {
    const debugElement = document.createElement('div');
    debugElement.className = 'mb-2 text-sm';
    debugElement.innerHTML = `<strong class="text-blue-300">${nodeId} - ${subnodeLabel}:</strong> ${info}`;
    debugContent.appendChild(debugElement);
    debugContent.scrollTop = debugContent.scrollHeight;
}

async function processInput(input) {
    debugContent.innerHTML = ''; // Clear previous debug info
    await activateNode('input');
    addDebugInfo('Input', '', `Received input: "${input}"`);

    for (const node of nodes.slice(1, -1)) { // Skip input and output nodes
        await activateNode(node.id);
        const activeSubnode = activateRandomSubnode(node.id);
        if (activeSubnode) {
            const debugInfo = await simulateProcessing(node.id, activeSubnode);
            addDebugInfo(node.id, activeSubnode, debugInfo);
        }
        await new Promise(resolve => setTimeout(resolve, 200)); // Small delay between nodes
    }

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ user_input: input }),
        });

        const data = await response.json();
        await activateNode('output');
        addDebugInfo('Output', '', `Generated response: "${data.response}"`);
        return data.response;
    } catch (error) {
        console.error('Error:', error);
        addDebugInfo('Output', '', 'An error occurred while processing the request.');
        return 'An error occurred while processing your request.';
    }
}

async function simulateProcessing(nodeId, subnodeLabel) {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, Math.random() * 500 + 500));

    // Generate some fake debug info based on the node and subnode
    const debugInfoMap = {
        'memory': {
            'Short-term Layer 1': 'Storing recent input',
            'Short-term Layer 2': 'Retrieving relevant memories',
            'Short-term Layer 3': 'Consolidating information',
            'Long-term Memories': 'Accessing past experiences',
            'Self Memories': 'Reflecting on personal knowledge'
        },
        'cognition': {
            'Thought Generation': 'Brainstorming ideas',
            'Analysis': 'Evaluating input context',
            'Reasoning': 'Applying logical inference',
            'Problem Solving': 'Developing solution strategies',
            'Meta Reasoning': 'Reflecting on thought processes'
        },
        'affect': {
            'Emotions': 'Assessing emotional context',
            'Personality Traits': 'Applying personality model',
            'Emotional Regulation': 'Balancing emotional response'
        },
        'context': {
            'Current Context': 'Evaluating conversation state',
            'Context Update': 'Incorporating new information',
            'Weight Management': 'Adjusting contextual priorities'
        },
        'safety': {
            'Content Moderation': 'Checking for inappropriate content',
            'Ethical Evaluation': 'Ensuring ethical response'
        }
    };

    return debugInfoMap[nodeId][subnodeLabel] || 'Processing data';
}

function addToConversationHistory(speaker, text) {
    const messageElement = document.createElement('div');
    messageElement.className = 'mb-2';
    messageElement.innerHTML = `<strong class="${speaker === 'You' ? 'text-blue-300' : 'text-blue-200'}">${speaker}:</strong> ${text}`;
    conversationHistory.appendChild(messageElement);
    scrollToBottom(conversationHistory);
}

function scrollToBottom(element) {
    element.scrollTop = element.scrollHeight;
}

sendButton.addEventListener('click', async () => {
    const input = userInput.value.trim();
    if (!input) return;

    addToConversationHistory('You', input);
    userInput.value = '';

    const response = await processInput(input);
    addToConversationHistory('BICA', response);
});

userInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendButton.click();
    }
});

createNodeTree();