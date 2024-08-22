const nodeNetwork = document.getElementById('node-network');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const conversationHistory = document.getElementById('conversation-history');
const debugContent = document.getElementById('debug-content');

const nodes = [
    { id: 'input', label: 'Input', group: 0, subnodes: [] },
    { id: 'memory', label: 'Memory', group: 1, subnodes: ['Short-term Layer 1', 'Short-term Layer 2', 'Short-term Layer 3', 'Long-term Memories', 'Self Memories'] },
    { id: 'cognition', label: 'Cognition', group: 2, subnodes: ['Thought Generation', 'Analysis', 'Reasoning', 'Problem Solving', 'Meta Reasoning'] },
    { id: 'affect', label: 'Affect', group: 3, subnodes: ['Emotions', 'Personality Traits', 'Emotional Regulation'] },
    { id: 'context', label: 'Context', group: 4, subnodes: ['Current Context', 'Context Update', 'Weight Management'] },
    { id: 'safety', label: 'Safety', group: 5, subnodes: ['Content Moderation', 'Ethical Evaluation'] },
    { id: 'output', label: 'Output', group: 6, subnodes: [] }
];

const links = [
    { source: 'input', target: 'memory' },
    { source: 'input', target: 'cognition' },
    { source: 'memory', target: 'cognition' },
    { source: 'memory', target: 'affect' },
    { source: 'cognition', target: 'affect' },
    { source: 'cognition', target: 'context' },
    { source: 'affect', target: 'context' },
    { source: 'context', target: 'safety' },
    { source: 'safety', target: 'output' }
];

let svg, link, node;

function createNodeNetwork() {
    const width = nodeNetwork.clientWidth;
    const height = nodeNetwork.clientHeight;

    svg = d3.select("#node-network")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .call(d3.zoom().on("zoom", zoomed));

    const g = svg.append("g");

    const allNodes = nodes.concat(nodes.flatMap(node =>
        node.subnodes.map(subnode => ({
            id: `${node.id}-${subnode}`,
            label: subnode,
            group: node.group,
            parent: node.id,
            isSubnode: true
        }))
    ));

    const allLinks = links.concat(nodes.flatMap(node =>
        node.subnodes.map(subnode => ({
            source: node.id,
            target: `${node.id}-${subnode}`
        }))
    ));

    const simulation = d3.forceSimulation(allNodes)
        .force("link", d3.forceLink(allLinks).id(d => d.id).distance(d => d.source.isSubnode || d.target.isSubnode ? 50 : 150))
        .force("charge", d3.forceManyBody().strength(d => d.isSubnode ? -100 : -500))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(d => d.isSubnode ? 20 : 50));

    link = g.append("g")
        .selectAll("line")
        .data(allLinks)
        .join("line")
        .attr("stroke", "#888")
        .attr("stroke-opacity", 0.6)
        .attr("stroke-width", d => d.source.isSubnode || d.target.isSubnode ? 1 : 2);

    node = g.append("g")
        .selectAll("g")
        .data(allNodes)
        .join("g")
        .call(d3.drag()
            .on("start", dragstarted)
            .on("drag", dragged)
            .on("end", dragended));

    node.append("circle")
        .attr("r", d => d.isSubnode ? 15 : 40)
        .attr("fill", d => {
            const baseColor = d3.rgb(d3.schemeCategory10[d.group]);
            return d.isSubnode ? baseColor.brighter(0.5).toString() : baseColor.toString();
        })
        .attr("data-id", d => d.id);

    node.append("text")
        .text(d => d.label)
        .attr("text-anchor", "middle")
        .attr("dy", ".35em")
        .attr("fill", "white")
        .style("font-size", d => d.isSubnode ? "8px" : "12px");

    simulation.on("tick", () => {
        link
            .attr("x1", d => d.source.x)
            .attr("y1", d => d.source.y)
            .attr("x2", d => d.target.x)
            .attr("y2", d => d.target.y);

        node
            .attr("transform", d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }

    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }

    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }

    function zoomed(event) {
        g.attr("transform", event.transform);
    }
}

function activateNode(nodeId, subnodeLabel = null) {
    const selector = subnodeLabel
        ? `circle[data-id="${nodeId}-${subnodeLabel}"]`
        : `circle[data-id="${nodeId}"]`;

    svg.selectAll(selector)
        .transition()
        .duration(300)
        .attr("r", d => d.isSubnode ? 20 : 50)
        .attr("fill", "yellow")
        .transition()
        .duration(300)
        .attr("r", d => d.isSubnode ? 15 : 40)
        .attr("fill", d => {
            const baseColor = d3.rgb(d3.schemeCategory10[d.group]);
            return d.isSubnode ? baseColor.brighter(0.5).toString() : baseColor.toString();
        });
}

function addDebugInfo(nodeId, subnodeLabel, info) {
    const debugElement = document.createElement('div');
    debugElement.className = 'mb-2 text-sm';
    debugElement.innerHTML = `<strong class="text-blue-300">${nodeId}${subnodeLabel ? ` - ${subnodeLabel}` : ''}:</strong> ${info}`;
    debugContent.appendChild(debugElement);
    debugContent.scrollTop = debugContent.scrollHeight;
}

async function processInput(input) {
    debugContent.innerHTML = '';
    await activateNode('input');
    addDebugInfo('Input', '', `Received input: "${input}"`);

    for (const node of nodes.slice(1, -1)) {
        await activateNode(node.id);
        const subnode = node.subnodes[Math.floor(Math.random() * node.subnodes.length)];
        if (subnode) {
            await activateNode(node.id, subnode);
            const debugInfo = await simulateProcessing(node.id, subnode);
            addDebugInfo(node.id, subnode, debugInfo);
        }
        await new Promise(resolve => setTimeout(resolve, 500));
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
    await new Promise(resolve => setTimeout(resolve, Math.random() * 500 + 500));

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
    conversationHistory.scrollTop = conversationHistory.scrollHeight;
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

createNodeNetwork();