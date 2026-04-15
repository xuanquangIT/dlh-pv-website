function updateKPIDom(data) {
    document.getElementById('kpi-valid').textContent = data.valid || '--';
    document.getElementById('kpi-valid-ratio').textContent = data.valid_ratio || '--';
    
    document.getElementById('kpi-warning').textContent = data.warning || '--';
    document.getElementById('kpi-warning-ratio').textContent = data.warning_ratio || '--';

    document.getElementById('kpi-invalid').textContent = data.invalid || '--';
    document.getElementById('kpi-invalid-ratio').textContent = data.invalid_ratio || '--';
}

function updateFacilityScoresDom(data) {
    const tbody = document.getElementById('facility-quality-table');
    tbody.innerHTML = '';
    
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;">No data available</td></tr>';
        return;
    }
    
    tbody.innerHTML = data.map(item => {
        let scoreBadgeClass = 'badge-success';
        const scoreVal = parseInt(item.score);
        if (scoreVal < 90) scoreBadgeClass = 'badge-warn';
        if (scoreVal < 80) scoreBadgeClass = 'badge-danger';

        return `<tr>
            <td>${item.facility}</td>
            <td>${item.valid}</td>
            <td>${item.warning}</td>
            <td>${item.invalid}</td>
            <td><span class="badge ${scoreBadgeClass}">${item.score}</span></td>
        </tr>`;
    }).join('');
}

function updateRecentIssuesDom(data) {
    const tbody = document.getElementById('recent-issues-table');
    tbody.innerHTML = '';
    
    if (!data || data.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;">No recent issues found</td></tr>';
        return;
    }
    
    tbody.innerHTML = data.map(item => {
        let severityClass = 'badge-warn';
        if (item.severity === 'GOOD') severityClass = 'badge-success';
        else if (item.severity === 'BAD') severityClass = 'badge-danger';
        
        return `<tr>
            <td>${item.time}</td>
            <td>${item.facility}</td>
            <td>${item.sensor}</td>
            <td>${item.issue}</td>
            <td>${item.affected}</td>
            <td><span class="badge ${severityClass}">${item.severity}</span></td>
            <td>${item.action}</td>
        </tr>`;
    }).join('');
}

function updateHeatmapDom(data) {
    const container = document.getElementById('heatmap-container');
    container.innerHTML = '';
    
    if (!data || data.length === 0) {
        container.innerHTML = '<div style="text-align:center;padding:20px;color:var(--text-sec);">No heatmap data available</div>';
        return;
    }
    
    const facilities = [...new Set(data.map(d => d.facility))].sort();
    const dates = [...new Set(data.map(d => d.date))].sort();
    
    const map = {};
    data.forEach(d => {
        if(!map[d.facility]) map[d.facility] = {};
        map[d.facility][d.date] = d.score;
    });
    
    let html = `<div style="display:flex; flex-direction:column; gap:6px; overflow-x:auto;">`;

    // Header dates
    html += `<div style="display:flex; gap:4px; margin-bottom: 4px;">
                <div style="width:100px; flex-shrink:0;"></div>`;
    dates.forEach(date => {
        const d = new Date(date).getDate();
        html += `<div style="width:28px; font-size:10px; text-align:center; color:var(--text-sec);">${d}</div>`;
    });
    html += `</div>`;
    
    // Rows
    facilities.forEach(fac => {
        html += `<div style="display:flex; gap:4px; align-items:center;">
            <div style="width:100px; font-size:11px; font-weight:500; flex-shrink:0; overflow:hidden; text-overflow:ellipsis;" title="${fac}">${fac}</div>`;
        dates.forEach(date => {
            const score = map[fac][date];
            let color = '#e2e8f0'; // distinct light grey for empty
            if (score !== undefined) {
                if (score >= 90) color = '#10b981'; // green
                else if (score >= 70) color = '#f59e0b'; // yellow
                else color = '#ef4444'; // red
            }
            const tooltip = score !== undefined ? `${fac} on ${date}: ${Number(score).toFixed(1)}% Valid` : `${fac} on ${date}: No Data`;
            html += `<div title="${tooltip}" style="width:28px; height:28px; background-color:${color}; border-radius:4px; flex-shrink:0;"></div>`;
        });
        html += `</div>`;
    });
    html += `</div>`;
    container.innerHTML = html;
}

async function fetchDataQualityData() {
    try {
        const [summaryRes, scoresRes, issuesRes, heatmapRes] = await Promise.all([
            fetch('/data-quality/summary'),
            fetch('/data-quality/facility-scores'),
            fetch('/data-quality/recent-issues'),
            fetch('/data-quality/heatmap-data')
        ]);        
        if (summaryRes.ok) {
            const summaryData = await summaryRes.json();
            updateKPIDom(summaryData);
        } else {
            console.error('Failed to fetch summary');
        }
        
        if (scoresRes.ok) {
            const scoresData = await scoresRes.json();
            updateFacilityScoresDom(scoresData);
        } else {
            console.error('Failed to fetch facility scores');
        }
        
        if (issuesRes.ok) {
            const issuesData = await issuesRes.json();
            updateRecentIssuesDom(issuesData);
        } else {
            console.error('Failed to fetch recent issues');
        }
        
        if (heatmapRes.ok) {
            const heatmapData = await heatmapRes.json();
            updateHeatmapDom(heatmapData);
        } else {
            console.error('Failed to fetch heatmap data');
        }
        
    } catch (error) {
        console.error('Error fetching data quality metrics:', error);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    fetchDataQualityData();
});
