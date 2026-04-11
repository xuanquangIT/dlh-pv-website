async function fetchPipelineData() {
    try {
        const [jobInfo, jobRuns] = await Promise.all([
            fetch('/data-pipeline/jobs').then(r => r.json()),
            fetch('/data-pipeline/jobs/runs').then(r => r.json())
        ]);
        
        renderJobInfo(jobInfo);
        renderRecentRuns(jobRuns);
        
        if (jobRuns.length > 0) {
            const latestRun = jobRuns[0];
            let tasks = latestRun.tasks || [];
            if(tasks.length === 0 && latestRun.run_id) {
                try {
                    const runDetails = await fetch(`/data-pipeline/jobs/runs/${latestRun.run_id}`).then(r => r.json());
                    tasks = runDetails.tasks || [];
                } catch(e) {
                    console.error("Could not fetch detailed run tasks", e);
                }
            }
            renderDAG(jobInfo.settings.tasks, tasks);
        }
    } catch (err) {
        console.error("Failed to load pipeline data", err);
    }
}

function renderJobInfo(jobInfo) {
    document.getElementById('job-name').textContent = jobInfo.settings.name || 'pv-lakehouse-incremental';
    if(jobInfo.settings.schedule) {
        document.getElementById('job-cron').textContent = jobInfo.settings.schedule.quartz_cron_expression || '0 0 3 * * ?';
    }
}

function renderRecentRuns(runs) {
    const tbody = document.getElementById('recent-runs-body');
    tbody.innerHTML = '';
    
    runs.forEach(run => {
        const tr = document.createElement('tr');
        
        const dateCell = document.createElement('td');
        dateCell.textContent = new Date(run.start_time).toLocaleString();
        
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
        tr.appendChild(stateCell);
        tr.appendChild(durationCell);
        tr.appendChild(actionsCell);
        
        tbody.appendChild(tr);
    });
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

function renderDAG(configuredTasks, runTasks) {
    // Map run tasks by task_key for quick status lookup
    const taskStatusMap = {};
    if(runTasks) {
        runTasks.forEach(t => {
            taskStatusMap[t.task_key] = t.state.result_state || t.state.life_cycle_state;
        });
    }

    const bronzeContainer = document.getElementById('layer-bronze');
    const silverContainer = document.getElementById('layer-silver');
    const goldContainer = document.getElementById('layer-gold');
    const mlContainer = document.getElementById('layer-ml');
    
    [bronzeContainer, silverContainer, goldContainer, mlContainer].forEach(el => el.innerHTML = '');

    configuredTasks.forEach(task => {
        const status = taskStatusMap[task.task_key] || 'PENDING';
        
        const taskEl = document.createElement('div');
        taskEl.className = 'task-node';
        taskEl.innerHTML = `
            <div class="task-status status-${status}" title="${status}"></div>
            <div class="task-name">${task.task_key}</div>
            <div class="task-duration">${status}</div>
        `;
        
        // Categorize based on prefix
        if (task.task_key.startsWith('bronze')) {
            bronzeContainer.appendChild(taskEl);
        } else if (task.task_key.startsWith('silver')) {
            silverContainer.appendChild(taskEl);
        } else if (task.task_key.startsWith('gold') || task.task_key.startsWith('forecast')) {
            if (task.task_key.includes('forecast_serving') || task.task_key.includes('diagnostics')) {
                mlContainer.appendChild(taskEl);
            } else {
                goldContainer.appendChild(taskEl);
            }
        } else {
            // default fallback to bronze or silver if setup
            if(task.task_key.includes('setup')) {
                if(task.task_key.includes('silver')) silverContainer.appendChild(taskEl);
                else if(task.task_key.includes('gold')) goldContainer.appendChild(taskEl);
                else bronzeContainer.appendChild(taskEl);
            }
        }
    });
}

document.addEventListener('DOMContentLoaded', () => {
    fetchPipelineData();
    setInterval(fetchPipelineData, 30000); // Polling every 30s
    
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
                const response = await fetch('/data-pipeline/jobs/run', { method: 'POST' });
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
});
