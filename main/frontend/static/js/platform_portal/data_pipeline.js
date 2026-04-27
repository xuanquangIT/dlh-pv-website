// ─── Active job state ─────────────────────────────────────────────────────────
const DEFAULT_JOB_ID = 28718564084384; // matches backend DEFAULT_JOB_ID
let currentJobId = DEFAULT_JOB_ID;

async function loadJobList() {
    const sel = document.getElementById('job-selector');
    if (!sel) return;
    try {
        const res = await fetch('/data-pipeline/jobs/list');
        if (!res.ok) {
            const err = await res.text();
            console.error(`[jobs/list] HTTP ${res.status}:`, err);
            sel.innerHTML = `<option value="${DEFAULT_JOB_ID}">pv-lakehouse-incremental</option>`;
            currentJobId = DEFAULT_JOB_ID;
            return;
        }
        const jobs = await res.json();
        if (!Array.isArray(jobs) || jobs.length === 0) {
            sel.innerHTML = `<option value="${DEFAULT_JOB_ID}">pv-lakehouse-incremental</option>`;
            currentJobId = DEFAULT_JOB_ID;
            return;
        }
        sel.innerHTML = jobs.map(j => {
            const paused = j.pause_status === 'PAUSED' ? ' [Paused]' : '';
            const selected = j.job_id === currentJobId ? ' selected' : '';
            return `<option value="${j.job_id}"${selected}>${j.name}${paused}</option>`;
        }).join('');
        // Sync currentJobId with what's actually selected
        currentJobId = parseInt(sel.value, 10) || DEFAULT_JOB_ID;
    } catch (err) {
        console.error('[jobs/list] Network error:', err);
        sel.innerHTML = `<option value="${DEFAULT_JOB_ID}">pv-lakehouse-incremental</option>`;
        currentJobId = DEFAULT_JOB_ID;
    }
}

async function fetchPipelineData() {
    const tbody = document.getElementById('recent-runs-body');
    const scheduleEl = document.getElementById('schedule-display');

    // 1. Fetch job info
    let jobInfo = null;
    try {
        const res = await fetch(`/data-pipeline/jobs?job_id=${currentJobId}`);
        if (!res.ok) {
            const detail = await res.text();
            console.error(`[jobs] HTTP ${res.status}:`, detail);
            if (scheduleEl) scheduleEl.textContent = `Error ${res.status} loading job info`;
        } else {
            jobInfo = await res.json();
            renderJobInfo(jobInfo);
        }
    } catch (err) {
        console.error('[jobs] Network error:', err);
        if (scheduleEl) scheduleEl.textContent = 'Network error loading job info';
    }

    // 2. Fetch runs
    let jobRuns = [];
    try {
        const res = await fetch(`/data-pipeline/jobs/runs?job_id=${currentJobId}&limit=15`);
        if (!res.ok) {
            const detail = await res.text();
            console.error(`[jobs/runs] HTTP ${res.status}:`, detail);
            if (tbody) tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--txt3);">Error ${res.status} loading runs</td></tr>`;
        } else {
            jobRuns = await res.json();
            renderRecentRuns(jobRuns);
        }
    } catch (err) {
        console.error('[jobs/runs] Network error:', err);
        if (tbody) tbody.innerHTML = `<tr><td colspan="5" style="text-align:center;color:var(--txt3);">Network error loading runs</td></tr>`;
    }

    // 3. Render DAG from latest run tasks
    if (jobInfo && jobRuns.length > 0) {
        const latestRun = jobRuns[0];
        let tasks = latestRun.tasks || [];
        if (tasks.length === 0 && latestRun.run_id) {
            try {
                const runDetails = await fetch(`/data-pipeline/jobs/runs/${latestRun.run_id}`).then(r => r.json());
                tasks = runDetails.tasks || [];
            } catch(e) {
                console.error('Could not fetch detailed run tasks', e);
            }
        }
        const configTasks = (jobInfo.settings && jobInfo.settings.tasks) || [];
        if (configTasks.length > 0) renderDAG(configTasks, tasks);
    }
}


const QUARTZ_DAY_LABEL_BY_NUMBER = {
    '1': 'Sunday',
    '2': 'Monday',
    '3': 'Tuesday',
    '4': 'Wednesday',
    '5': 'Thursday',
    '6': 'Friday',
    '7': 'Saturday'
};

const QUARTZ_DAY_NUMBER_BY_NAME = {
    SUN: '1',
    MON: '2',
    TUE: '3',
    WED: '4',
    THU: '5',
    FRI: '6',
    SAT: '7'
};

function normalizeQuartzDayForPicker(rawValue) {
    const token = String(rawValue || '').trim().toUpperCase();
    if (!token || token === '*' || token === '?') return '1';

    if (QUARTZ_DAY_NUMBER_BY_NAME[token]) {
        return QUARTZ_DAY_NUMBER_BY_NAME[token];
    }

    const numeric = Number.parseInt(token, 10);
    if (Number.isFinite(numeric)) {
        if (numeric === 0) return '1';
        if (numeric >= 1 && numeric <= 7) return String(numeric);
    }

    return '1';
}

function getQuartzDayLabel(dayOfWeek) {
    const normalized = normalizeQuartzDayForPicker(dayOfWeek);
    return QUARTZ_DAY_LABEL_BY_NUMBER[normalized] || 'Sunday';
}

function getFirstCronToken(value) {
    return String(value || '').split(',')[0].split('-')[0].trim();
}

function parseQuartzToPicker(cronExpr) {
    const parts = cronExpr.trim().split(/\s+/);
    const minute = (parts[1] || '0').trim();
    const hour = (parts[2] || '0').trim();
    const dayOfMonth = (parts[3] || '*').trim();
    const dayOfWeek = (parts[5] || '*').trim();
    
    let frequency = 'daily';
    if (dayOfWeek !== '*' && dayOfWeek !== '?') frequency = 'weekly';
    else if (dayOfMonth !== '*' && dayOfMonth !== '?') frequency = 'monthly';
    
    return { 
        minute: isNaN(minute) ? '0' : minute,
        hour: isNaN(hour) ? '0' : hour,
        dayOfMonth: dayOfMonth === '*' || dayOfMonth === '?' ? '1' : dayOfMonth,
        dayOfWeek: normalizeQuartzDayForPicker(dayOfWeek),
        frequency 
    };
}

function generateQuartzCron(frequency, hour, minute, dayOfWeek = '*', dayOfMonth = '*') {
    hour = String(hour).padStart(2, '0');
    minute = String(minute).padStart(2, '0');
    const normalizedDayOfWeek = normalizeQuartzDayForPicker(dayOfWeek);
    
    if (frequency === 'weekly') {
        return `0 ${minute} ${hour} ? * ${normalizedDayOfWeek}`;
    } else if (frequency === 'monthly') {
        return `0 ${minute} ${hour} ${dayOfMonth} * ?`;
    }
    return `0 ${minute} ${hour} * * ?`;
}

function formatScheduleReadable(cron, timezone, frequency, hour, minute, dayOfWeek) {
    hour = String(hour).padStart(2, '0');
    minute = String(minute).padStart(2, '0');
    
    let freq_str = 'Every day';
    if (frequency === 'weekly') {
        freq_str = `Every ${getQuartzDayLabel(dayOfWeek)}`;
    } else if (frequency === 'monthly') {
        freq_str = 'Every month';
    }
    
    return `${freq_str} at ${hour}:${minute} (${timezone})`;
}

function renderJobInfo(jobInfo) {
    document.getElementById('job-name').textContent = jobInfo.settings.name || 'pv-lakehouse-incremental';
    if(jobInfo.settings.schedule) {
        const schedule = jobInfo.settings.schedule;
        const cron = schedule.quartz_cron_expression || '0 0 3 * * ?';
        const timezone = schedule.timezone_id || 'UTC';

        const parsed = parseQuartzToPicker(cron);
        const readableSchedule = formatScheduleReadable(cron, timezone, parsed.frequency, parsed.hour, parsed.minute, parsed.dayOfWeek);
        
        const displayEl = document.getElementById('schedule-display');
        if (displayEl) {
            displayEl.textContent = readableSchedule;
        }

        const frequencySelect = document.getElementById('schedule-frequency');
        const hourInput = document.getElementById('schedule-hour');
        const minuteInput = document.getElementById('schedule-minute');
        const dayOfWeekSelect = document.getElementById('schedule-day-of-week');
        const dayOfMonthSelect = document.getElementById('schedule-day-of-month');
        const timezoneSelect = document.getElementById('schedule-timezone');

        if (frequencySelect && document.activeElement !== frequencySelect) {
            frequencySelect.value = parsed.frequency;
            updateScheduleUIVisibility();
        }
        if (hourInput && document.activeElement !== hourInput) hourInput.value = String(parsed.hour).padStart(2, '0');
        if (minuteInput && document.activeElement !== minuteInput) minuteInput.value = String(parsed.minute).padStart(2, '0');
        
        const dowValue = normalizeQuartzDayForPicker(getFirstCronToken(parsed.dayOfWeek));
        if (dayOfWeekSelect && document.activeElement !== dayOfWeekSelect && dowValue !== '*' && dowValue !== '?') {
            dayOfWeekSelect.value = dowValue;
        }
        
        const domValue = getFirstCronToken(parsed.dayOfMonth);
        if (dayOfMonthSelect && document.activeElement !== dayOfMonthSelect && domValue !== '*' && domValue !== '?') {
            dayOfMonthSelect.value = domValue;
        }
        
        if (timezoneSelect && document.activeElement !== timezoneSelect) timezoneSelect.value = timezone;
    } else {
        const displayEl = document.getElementById('schedule-display');
        if (displayEl) displayEl.textContent = 'No schedule configured';
    }
}

function updateScheduleUIVisibility() {
    const frequency = document.getElementById('schedule-frequency')?.value || 'daily';
    const dayRow = document.getElementById('schedule-day-row');
    const dateRow = document.getElementById('schedule-date-row');
    
    if (dayRow) dayRow.style.display = frequency === 'weekly' ? 'flex' : 'none';
    if (dateRow) dateRow.style.display = frequency === 'monthly' ? 'flex' : 'none';
}

function renderRecentRuns(runs) {
    const tbody = document.getElementById('recent-runs-body');
    tbody.innerHTML = '';

    if (!runs || runs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--txt3);padding:20px;">No runs found for this job</td></tr>';
        return;
    }

    runs.forEach(run => {
        const tr = document.createElement('tr');
        
        const dateCell = document.createElement('td');
        dateCell.textContent = new Date(run.start_time).toLocaleString();
        
        const launchedCell = document.createElement('td');
        const launchType = getLaunchType(run);
        launchedCell.textContent = launchType;
        launchedCell.className = 'launched-cell';
        if (launchType === 'Manually') {
            launchedCell.classList.add('launched-manually');
        } else if (launchType === 'By scheduler') {
            launchedCell.classList.add('launched-scheduler');
        }
        
        const stateCell = document.createElement('td');
        const stateRaw = run.state.result_state || run.state.life_cycle_state;
        stateCell.innerHTML = `<span class="badge ${getBadgeClass(stateRaw)}">${stateRaw}</span>`;
        
        const durationCell = document.createElement('td');
        const durationMs = run.run_duration || run.execution_duration || 0;
        durationCell.textContent = formatDuration(durationMs);
        
        const actionsCell = document.createElement('td');
        if (stateRaw === 'RUNNING' || stateRaw === 'PENDING') {
            const cancelBtn = document.createElement('button');
            cancelBtn.className = 'btn-danger-sm';
            cancelBtn.textContent = 'Cancel';
            cancelBtn.onclick = async () => {
                cancelBtn.disabled = true;
                cancelBtn.textContent = 'Canceling...';
                try {
                    const res = await fetch(`/data-pipeline/jobs/runs/${run.run_id}/cancel`, { method: 'POST' });
                    if(res.ok) {
                        alert('Run cancellation requested.');
                        fetchPipelineData();
                    } else {
                        alert('Failed to cancel run.');
                    }
                } catch(e) {
                    alert('Error canceling run.');
                } finally {
                    cancelBtn.disabled = false;
                    cancelBtn.textContent = 'Cancel';
                }
            };
            actionsCell.appendChild(cancelBtn);
        }

        tr.appendChild(dateCell);
        tr.appendChild(launchedCell);
        tr.appendChild(stateCell);
        tr.appendChild(durationCell);
        tr.appendChild(actionsCell);
        
        tbody.appendChild(tr);
    });
}

function getLaunchType(run) {
    // Databricks trigger field is a string:
    // - "ONE_TIME" = Manual trigger (user clicked "Run Now")
    // - "PERIODIC" = Scheduled trigger
    
    if (run.trigger) {
        if (run.trigger === 'ONE_TIME') {
            return 'Manually';
        } else if (run.trigger === 'PERIODIC') {
            return 'By scheduler';
        }
    }
    
    // Fallback
    return 'Unknown';
}

function formatDuration(ms) {
    if (!ms) return '0s';
    const totalSeconds = Math.floor(ms / 1000);
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = totalSeconds % 60;
    
    let parts = [];
    if (h > 0) parts.push(`${h}h`);
    if (m > 0 || h > 0) parts.push(`${m}m`);
    parts.push(`${s}s`);
    
    return parts.join(' ');
}

function getBadgeClass(state) {
    switch(state) {
        case 'SUCCESS': return 'badge-success';
        case 'FAILED': return 'badge-danger';
        case 'RUNNING': return 'badge-info';
        case 'SKIPPED': return 'badge-warn';
        default: return 'badge-default';
    }
}

/**
 * Render the pipeline DAG dynamically.
 *
 * Strategy:
 *  1. Compute topological "depth" for every task (0 = no deps, 1 = deps on 0, etc.)
 *  2. Group tasks by depth → each depth becomes one horizontal layer.
 *  3. Assign a human-readable label from the task_key prefix if possible
 *     (bronze/silver/gold/ml), otherwise "Stage N".
 *  4. Inject layers into #dag-dynamic, clearing previous content first.
 */
function renderDAG(configuredTasks, runTasks) {
    const dagEl = document.getElementById('dag-dynamic');
    if (!dagEl) return;
    dagEl.innerHTML = '';

    if (!configuredTasks || configuredTasks.length === 0) {
        dagEl.innerHTML = '<div style="text-align:center;color:var(--txt3);padding:24px;">No task configuration available for this job.</div>';
        return;
    }

    // ── Build status map from latest run ──────────────────────────────────
    const taskStatusMap = {};
    if (runTasks) {
        runTasks.forEach(t => {
            taskStatusMap[t.task_key] = t.state.result_state || t.state.life_cycle_state;
        });
    }

    // ── Compute topological depth (BFS) ──────────────────────────────────
    const depMap = {};  // task_key → Set of dependency task_keys
    configuredTasks.forEach(t => {
        depMap[t.task_key] = new Set((t.depends_on || []).map(d => d.task_key));
    });

    const depth = {};
    const queue = [];
    configuredTasks.forEach(t => {
        if (depMap[t.task_key].size === 0) {
            depth[t.task_key] = 0;
            queue.push(t.task_key);
        }
    });

    // Iterative BFS — safe even with deep DAGs
    while (queue.length > 0) {
        const current = queue.shift();
        configuredTasks.forEach(t => {
            if (depMap[t.task_key].has(current)) {
                const newDepth = (depth[current] ?? 0) + 1;
                if (depth[t.task_key] === undefined || depth[t.task_key] < newDepth) {
                    depth[t.task_key] = newDepth;
                }
                // Check if all deps resolved
                const allDepsResolved = [...depMap[t.task_key]].every(d => depth[d] !== undefined);
                if (allDepsResolved && !queue.includes(t.task_key)) {
                    queue.push(t.task_key);
                }
            }
        });
    }

    // ── Group tasks by depth ──────────────────────────────────────────────
    const byDepth = {};
    configuredTasks.forEach(t => {
        const d = depth[t.task_key] ?? 0;
        if (!byDepth[d]) byDepth[d] = [];
        byDepth[d].push(t);
    });

    const LAYER_COLORS = {
        bronze: { cls: 'layer-bronze', label: 'BRONZE' },
        silver: { cls: 'layer-silver', label: 'SILVER' },
        gold:   { cls: 'layer-gold',   label: 'GOLD'   },
        ml:     { cls: 'layer-ml',     label: 'ML & SERVING' },
        forecast: { cls: 'layer-gold', label: 'GOLD'   },
    };

    function getLayerMeta(tasks, depthIdx) {
        // Take the majority prefix from this layer's tasks
        const prefixCounts = {};
        tasks.forEach(t => {
            const key = t.task_key.toLowerCase();
            for (const prefix of Object.keys(LAYER_COLORS)) {
                if (key.startsWith(prefix)) {
                    prefixCounts[prefix] = (prefixCounts[prefix] || 0) + 1;
                    break;
                }
            }
        });
        const dominant = Object.entries(prefixCounts).sort((a,b) => b[1]-a[1])[0]?.[0];
        return LAYER_COLORS[dominant] || { cls: `layer-stage`, label: `Stage ${depthIdx + 1}` };
    }

    // ── Render layers ─────────────────────────────────────────────────────
    const sortedDepths = Object.keys(byDepth).map(Number).sort((a,b) => a-b);
    sortedDepths.forEach(d => {
        const tasks = byDepth[d];
        const meta  = getLayerMeta(tasks, d);

        const layerEl = document.createElement('div');
        layerEl.className = `dag-layer ${meta.cls}`;

        const labelEl = document.createElement('div');
        labelEl.style.cssText = 'font-weight:600; min-width:120px;';
        labelEl.textContent = meta.label;
        layerEl.appendChild(labelEl);

        const nodesEl = document.createElement('div');
        nodesEl.style.cssText = 'display:flex; gap:16px; flex-wrap:wrap;';

        tasks.forEach(task => {
            const status  = taskStatusMap[task.task_key] || 'PENDING';
            const deps    = (task.depends_on || []).map(dep => dep.task_key);
            const depText = deps.length === 0 ? '(starts first)' : `← ${deps.join(', ')}`;

            const taskEl = document.createElement('div');
            taskEl.className = 'task-node';
            taskEl.setAttribute('data-task-key', task.task_key);
            taskEl.innerHTML = `
                <div class="task-status status-${status}" title="${status}"></div>
                <div class="task-name">${task.task_key}</div>
                <div class="task-order">#${d + 1}</div>
                <div class="task-deps" title="Dependencies: ${depText}">${depText}</div>
                <div class="task-duration">${status}</div>
            `;
            nodesEl.appendChild(taskEl);
        });

        layerEl.appendChild(nodesEl);
        dagEl.appendChild(layerEl);
    });
}


document.addEventListener('DOMContentLoaded', () => {
    // Load job list first, then fetch data for the selected job
    loadJobList().then(() => {
        fetchPipelineData();
    });
    setInterval(fetchPipelineData, 30000); // Polling every 30s

    // Job selector change
    const jobSel = document.getElementById('job-selector');
    if (jobSel) {
        jobSel.addEventListener('change', () => {
            currentJobId = parseInt(jobSel.value, 10);
            // Clear stale content immediately
            const dagEl = document.getElementById('dag-dynamic');
            if (dagEl) dagEl.innerHTML = '<div style="text-align:center;color:var(--txt3);padding:24px;">Loading...</div>';
            const tbody = document.getElementById('recent-runs-body');
            if (tbody) tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;">Loading...</td></tr>';
            const scheduleEl = document.getElementById('schedule-display');
            if (scheduleEl) scheduleEl.textContent = 'Loading...';
            fetchPipelineData();
        });
    }
    
    // Tab Switcher Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active from all
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Set active
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    // Attach RUN NOW event
    const runBtn = document.getElementById('pipeline-run-now');
    if(runBtn) {
        runBtn.addEventListener('click', async () => {
            runBtn.disabled = true;
            runBtn.textContent = 'Triggering...';
            try {
                const response = await fetch(`/data-pipeline/jobs/run?job_id=${currentJobId}`, { method: 'POST' });
                if(response.ok) {
                    alert('Pipeline triggered successfully!');
                    fetchPipelineData(); // Refresh immediately
                } else {
                    alert('Failed to trigger pipeline.');
                }
            } catch (err) {
                console.error("Trigger failed", err);
                alert('Failed to trigger pipeline.');
            } finally {
                runBtn.disabled = false;
                runBtn.textContent = 'Run Now';
            }
        });
    }

    // Toggle Schedule Form
    const scheduleForm = document.getElementById('schedule-form');
    const toggleBtn = document.getElementById('toggle-schedule-form');
    const cancelBtn = document.getElementById('cancel-schedule-form');
    
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            if (scheduleForm.style.display === 'none') {
                scheduleForm.style.display = 'flex';
                toggleBtn.textContent = 'Hide Schedule';
                toggleBtn.classList.add('active');
            } else {
                scheduleForm.style.display = 'none';
                toggleBtn.textContent = 'Edit Schedule';
                toggleBtn.classList.remove('active');
            }
        });
    }
    
    if (cancelBtn) {
        cancelBtn.addEventListener('click', () => {
            scheduleForm.style.display = 'none';
            if (toggleBtn) {
                toggleBtn.textContent = 'Edit Schedule';
                toggleBtn.classList.remove('active');
            }
        });
    }

    const frequencySelect = document.getElementById('schedule-frequency');
    if (frequencySelect) {
        frequencySelect.addEventListener('change', updateScheduleUIVisibility);
        updateScheduleUIVisibility();
    }

    if (scheduleForm) {
        scheduleForm.addEventListener('submit', async (event) => {
            event.preventDefault();

            const saveBtn = document.getElementById('pipeline-save-schedule');
            const frequencySelect = document.getElementById('schedule-frequency');
            const hourInput = document.getElementById('schedule-hour');
            const minuteInput = document.getElementById('schedule-minute');
            const dayOfWeekSelect = document.getElementById('schedule-day-of-week');
            const dayOfMonthSelect = document.getElementById('schedule-day-of-month');
            const timezoneSelect = document.getElementById('schedule-timezone');

            const frequency = frequencySelect?.value || 'daily';
            const hour = parseInt(hourInput?.value || '0', 10);
            const minute = parseInt(minuteInput?.value || '0', 10);
            const dayOfWeek = dayOfWeekSelect?.value || '1';
            const dayOfMonth = dayOfMonthSelect?.value || '1';
            const timezone = (timezoneSelect?.value || 'UTC').trim();

            const generatedCron = generateQuartzCron(frequency, hour, minute, dayOfWeek, dayOfMonth);

            const payload = {
                quartz_cron_expression: generatedCron,
                timezone_id: timezone || null,
                pause_status: 'UNPAUSED'
            };

            if (saveBtn) {
                saveBtn.disabled = true;
                saveBtn.textContent = 'Saving...';
            }

            try {
                const response = await fetch(`/data-pipeline/jobs/schedule?job_id=${currentJobId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    const message = errorData.detail || 'Failed to update schedule.';
                    throw new Error(message);
                }

                alert('Schedule updated successfully.');
                await fetchPipelineData();
            } catch (error) {
                console.error('Schedule update failed', error);
                alert(error.message || 'Failed to update schedule.');
            } finally {
                if (saveBtn) {
                    saveBtn.disabled = false;
                    saveBtn.textContent = 'Save Schedule';
                }
            }
        });
    }
});
