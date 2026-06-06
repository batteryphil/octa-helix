const API = "http://localhost:8000";

const ARM_LABELS = [
    "General Language","Symbolic Math","Logical Reasoning","Code Syntax",
    "Factual Recall","Summarization","Creative Writing","Instruction Following",
    "Analogical Reasoning","Causal Inference","Spatial Reasoning",
    "Temporal Reasoning","Ethical Judgment","Multilingual Bridge",
    "Meta-Cognition","Synthesis",
];

let currentMode     = 'fast';    // 'fast' | 'deep'
let isGenerating    = false;
let showThinking    = true;
let history         = [];        // [{role, content}]

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
    buildArmLegend();
    buildArmBars([]);
    checkHealth();
    setInterval(checkHealth, 8000);
});

// ── Mode toggle ───────────────────────────────────────────────────────────────
function setMode(mode) {
    currentMode = mode;
    document.getElementById('btn-fast').classList.toggle('active', mode === 'fast');
    document.getElementById('btn-deep').classList.toggle('active', mode === 'deep');
    document.getElementById('send-icon').textContent  = mode === 'fast' ? '⚡' : '🧠';
    document.getElementById('mode-desc').textContent  =
        mode === 'fast'
            ? 'Single-pass generation. Fast and direct.'
            : 'Each arm reasons independently, then synthesizes. Slower but deeper.';
    document.getElementById('think-vis-toggle').style.display =
        mode === 'deep' ? 'flex' : 'none';
}

function toggleThinkingVisibility() {
    showThinking = document.getElementById('show-thinking').checked;
    // Toggle visibility of all existing think panels
    document.querySelectorAll('.think-panel').forEach(el => {
        el.style.display = showThinking ? 'block' : 'none';
    });
}

// ── Arm bars ──────────────────────────────────────────────────────────────────
function buildArmLegend() {
    document.getElementById('arm-legend').innerHTML = ARM_LABELS.map((l,i) => `
        <div class="arm-legend-item">
            <span class="arm-num">${String(i).padStart(2,'0')}</span>
            <span>${l}</span>
        </div>`).join('');
}

function buildArmBars(weights) {
    const el = document.getElementById('arm-bars');
    if (!weights || !weights.length) {
        el.innerHTML = ARM_LABELS.map((l,i) => `
            <div class="arm-bar-row" id="arm-row-${i}">
                <span class="arm-bar-label">${l}</span>
                <div class="arm-bar-track"><div class="arm-bar-fill" style="width:${(100/16).toFixed(1)}%"></div></div>
                <span class="arm-weight-label">—</span>
            </div>`).join('');
        return;
    }
    const maxW = Math.max(...weights, 0.001);
    const topIdx = weights.indexOf(Math.max(...weights));
    el.innerHTML = ARM_LABELS.map((l,i) => {
        const w   = weights[i] || 0;
        const pct = ((w / maxW) * 100).toFixed(1);
        const isTop    = i === topIdx;
        const isActive = w > (1/16) * 1.2;
        return `
            <div class="arm-bar-row ${isTop?'top-arm':''} ${isActive?'active':''}" id="arm-row-${i}">
                <span class="arm-bar-label" title="Arm ${i}: ${l}">${l}</span>
                <div class="arm-bar-track"><div class="arm-bar-fill" style="width:${pct}%"></div></div>
                <span class="arm-weight-label">${w.toFixed(3)}</span>
            </div>`;
    }).join('');

    // Entropy badge
    if (weights.length) {
        const ent = -(weights.map(w => {
            const p = Math.max(w, 1e-9);
            return p * Math.log(p);
        }).reduce((a,b)=>a+b, 0));
        const maxEnt = Math.log(weights.length);
        const pct = (ent / maxEnt * 100).toFixed(0);
        const badge = document.getElementById('arm-entropy-badge');
        badge.textContent  = `${pct}% diverse`;
        badge.style.color  = ent > maxEnt * 0.6 ? 'var(--accent)' : 'var(--warn)';
    }
}

function updateMetrics(armInfo) {
    if (!armInfo) return;
    const ent = armInfo.entropy ?? armInfo.arm_info?.entropy;
    const ipc = armInfo.ipc_res_gate ?? armInfo.arm_info?.ipc_res_gate;
    const top = armInfo.top_arms?.[0] ?? armInfo.arm_info?.top_arms?.[0];
    if (ent !== undefined) document.getElementById('entropy-val').textContent = (+ent).toFixed(3);
    if (ipc !== undefined) document.getElementById('ipc-val').textContent     = (+ipc).toFixed(3);
    if (top)               document.getElementById('top-arm-val').textContent  = top.label ?? `Arm ${top.arm}`;
}

// ── Health ────────────────────────────────────────────────────────────────────
async function checkHealth() {
    try {
        const d = await (await fetch(`${API}/health`)).json();
        const dot = document.getElementById('pulse-dot');
        const hdr = document.getElementById('header-status');
        const sv  = document.getElementById('status-val');
        if (d.status === 'ready') {
            dot.className = 'pulse-dot';
            hdr.textContent = `Phase: ${(d.phase||'?').toUpperCase()}  ·  Step: ${(d.training_step||0).toLocaleString()}  ·  VRAM: ${d.vram_gb}GB`;
            sv.textContent  = 'Ready'; sv.style.color = 'var(--accent)';
        } else {
            dot.className = 'pulse-dot loading';
            hdr.textContent = 'Loading model...';
            sv.textContent  = 'Loading'; sv.style.color = 'var(--warn)';
        }
    } catch {
        document.getElementById('pulse-dot').className = 'pulse-dot error';
        document.getElementById('header-status').textContent = 'Server offline — run: python3 titan_server.py';
        document.getElementById('status-val').textContent = 'Offline';
        document.getElementById('status-val').style.color = 'var(--danger)';
    }
}

// ── Message rendering ─────────────────────────────────────────────────────────
function buildPrompt(text) {
    let p = history.map(t =>
        t.role === 'user' ? `User: ${t.content}\n` : `Assistant: ${t.content}\n`
    ).join('');
    return p + `User: ${text}\nAssistant: `;
}

function addUserMsg(text) {
    const c = document.getElementById('messages');
    const d = document.createElement('div');
    d.className = 'message user-msg';
    d.innerHTML = `<div class="message-role">You</div>
                   <div class="message-content">${escapeHtml(text)}</div>`;
    c.appendChild(d); c.scrollTop = c.scrollHeight;
}

function createTitanMsg() {
    const c = document.getElementById('messages');
    const wrapper = document.createElement('div');
    wrapper.className = 'message titan-msg';

    // Think panel (hidden until thoughts arrive)
    const thinkPanel = document.createElement('div');
    thinkPanel.className = 'think-panel';
    thinkPanel.style.display = (showThinking && currentMode === 'deep') ? 'block' : 'none';

    const thinkHeader = document.createElement('div');
    thinkHeader.className = 'think-header';
    thinkHeader.innerHTML = `
        <span class="think-spinner" id="think-spinner">⟳</span>
        <span class="think-title">Reasoning arms thinking...</span>
        <button class="think-toggle-btn" onclick="this.closest('.think-panel').classList.toggle('collapsed')">▲</button>`;

    const thinkBody = document.createElement('div');
    thinkBody.className = 'think-body';
    thinkBody.id = 'think-body-' + Date.now();

    thinkPanel.appendChild(thinkHeader);
    thinkPanel.appendChild(thinkBody);

    const roleEl    = document.createElement('div');
    roleEl.className = 'message-role';
    roleEl.innerHTML = '🧠 Titan';

    const contentEl = document.createElement('div');
    contentEl.className = 'message-content';

    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    contentEl.appendChild(cursor);

    const chipArea = document.createElement('div');
    chipArea.className = 'arm-chips';

    wrapper.appendChild(thinkPanel);
    wrapper.appendChild(roleEl);
    wrapper.appendChild(contentEl);
    wrapper.appendChild(chipArea);
    c.appendChild(wrapper);
    c.scrollTop = c.scrollHeight;

    return { wrapper, thinkPanel, thinkBody, thinkHeader, contentEl, cursor, chipArea };
}

function addThought(thinkBody, thought) {
    const w = thought.weight || 0;
    const bar = '█'.repeat(Math.round(w * 30));
    const el = document.createElement('div');
    el.className = 'thought-block';
    el.innerHTML = `
        <div class="thought-header">
            <span class="thought-arm">Arm ${String(thought.arm).padStart(2,'0')}</span>
            <span class="thought-label">${escapeHtml(thought.label)}</span>
            <span class="thought-weight">${(w*100).toFixed(1)}% ${bar}</span>
        </div>
        <div class="thought-text">${escapeHtml(thought.thought || '…')}</div>`;
    thinkBody.appendChild(el);
    thinkBody.scrollTop = thinkBody.scrollHeight;
}

function finalizeThinkPanel(thinkHeader, count) {
    const spinner = thinkHeader.querySelector('.think-spinner');
    if (spinner) spinner.textContent = '✅';
    const title = thinkHeader.querySelector('.think-title');
    if (title) title.textContent = `${count} arms reasoned — synthesizing response`;
}

function addArmChips(chipArea, topArms) {
    chipArea.innerHTML = '';
    (topArms || []).slice(0, 6).forEach((a, i) => {
        const chip = document.createElement('span');
        chip.className = `arm-chip ${i===0?'top':''}`;
        chip.textContent = `${a.label||`Arm ${a.arm}`} ${((a.weight||0)*100).toFixed(1)}%`;
        chipArea.appendChild(chip);
    });
}

// ── Send ──────────────────────────────────────────────────────────────────────
function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
}

async function sendMessage() {
    if (isGenerating) return;
    const input    = document.getElementById('user-input');
    const userText = input.value.trim();
    if (!userText) return;

    const temp = parseFloat(document.getElementById('temp-slider').value);
    const topP = parseFloat(document.getElementById('topp-slider').value);

    input.value = '';
    isGenerating = true;
    document.getElementById('send-btn').disabled = true;
    document.getElementById('send-label').textContent = '…';

    addUserMsg(userText);
    history.push({role:'user', content: userText});

    const { wrapper, thinkPanel, thinkBody, thinkHeader, contentEl, cursor, chipArea } = createTitanMsg();
    const c = document.getElementById('messages');

    const body = JSON.stringify({
        prompt: buildPrompt(userText),
        max_new_tokens: 512,
        temperature: temp, top_p: topP,
        show_arm_weights: true,
        thought_tokens: 60,
    });

    let fullText     = '';
    let lastWeights  = [];
    let thoughtCount = 0;

    try {
        const endpoint = currentMode === 'deep' ? '/stream_deep' : '/stream';
        const res = await fetch(`${API}${endpoint}`, {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body,
        });

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let   buf     = '';

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buf += decoder.decode(value, {stream: true});
            const lines = buf.split('\n');
            buf = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const ev = JSON.parse(line.slice(6));

                    // Deep think events
                    if (ev.type === 'thinking_start') {
                        thinkPanel.style.display = showThinking ? 'block' : 'none';
                    }
                    else if (ev.type === 'thought') {
                        thoughtCount++;
                        addThought(thinkBody, ev);
                        c.scrollTop = c.scrollHeight;
                    }
                    else if (ev.type === 'thinking_done') {
                        finalizeThinkPanel(thinkHeader, ev.thought_count || thoughtCount);
                    }
                    // Token events (both modes)
                    else if (ev.type === 'token' || ev.token !== undefined) {
                        const tok = ev.token || ev.token;
                        if (tok) {
                            fullText += tok;
                            contentEl.textContent = fullText;
                            contentEl.appendChild(cursor);
                            c.scrollTop = c.scrollHeight;
                        }
                        const info = ev.arm_info || {};
                        if (info.arm_weights?.length) {
                            lastWeights = info.arm_weights;
                            buildArmBars(lastWeights);
                            updateMetrics(info);
                        }
                    }
                    // Done events
                    else if (ev.type === 'done' || ev.done) {
                        const topArms = ev.top_arms || [];
                        cursor.remove();
                        contentEl.textContent = fullText.replace('<|endoftext|>','').trim();
                        addArmChips(chipArea, topArms);
                        if (lastWeights.length) buildArmBars(lastWeights);
                    }
                } catch {}
            }
        }

    } catch (err) {
        cursor.remove();
        contentEl.textContent = `[Connection error: ${err.message}]\nIs titan_server.py running on port 8000?`;
    }

    history.push({role:'titan', content: fullText.trim()});
    isGenerating = false;
    document.getElementById('send-btn').disabled = false;
    document.getElementById('send-label').textContent = currentMode === 'deep' ? 'Think' : 'Send';
    c.scrollTop = c.scrollHeight;
}

function clearChat() {
    history = [];
    document.getElementById('messages').innerHTML = `
        <div class="message system-msg">
            <div class="message-content"><p>🧠 Chat cleared.</p></div>
        </div>`;
    buildArmBars([]);
}

function escapeHtml(s) {
    return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;')
                         .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
