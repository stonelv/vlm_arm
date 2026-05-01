let currentFilename = null;
let currentTaskId = null;
let currentExecutionPlan = null;
let chatMessages = [];

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('imageInput').addEventListener('change', handleImageUpload);
    document.getElementById('taskInput').addEventListener('input', updateAnalyzeButton);
    
    updateTaskStatus('pending', '等待任务');
});

function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('image', file);
    
    const preview = document.getElementById('imagePreview');
    preview.innerHTML = '<div class="text-center"><div class="spinner-border text-primary" role="status"></div><p class="mt-2">上传中...</p></div>';
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentFilename = data.filename;
            
            preview.innerHTML = `<img src="${data.image_url}" alt="上传的图像" class="img-fluid rounded">`;
            
            addChatMessage('system', `图像已上传: ${data.image_info.width}x${data.image_info.height}`);
            updateAnalyzeButton();
        } else {
            preview.innerHTML = '<div class="text-center text-danger"><i class="bi bi-exclamation-triangle display-4"></i><p class="mt-2">上传失败: ' + data.error + '</p></div>';
        }
    })
    .catch(error => {
        preview.innerHTML = '<div class="text-center text-danger"><i class="bi bi-exclamation-triangle display-4"></i><p class="mt-2">上传出错</p></div>';
        console.error('Upload error:', error);
    });
}

function updateAnalyzeButton() {
    const btn = document.getElementById('analyzeBtn');
    const hasImage = currentFilename !== null;
    const hasInstruction = document.getElementById('taskInput').value.trim().length > 0;
    
    btn.disabled = !(hasImage && hasInstruction);
}

function analyzeTask() {
    const instruction = document.getElementById('taskInput').value.trim();
    
    if (!currentFilename || !instruction) {
        showAlert('请先上传图像并输入指令', 'warning');
        return;
    }
    
    addChatMessage('user', instruction);
    updateTaskStatus('planning', '正在分析任务...');
    
    document.getElementById('analyzeBtn').disabled = true;
    document.getElementById('analyzeBtn').innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>分析中...';
    
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            instruction: instruction,
            filename: currentFilename
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentTaskId = data.task_id;
            currentExecutionPlan = data.execution_plan;
            
            displayDetections(data.detections, data.detections_summary);
            displayExecutionPlan(data.execution_plan);
            
            document.getElementById('planCard').style.display = 'block';
            document.getElementById('exportCard').style.display = 'block';
            
            updateTaskStatus('awaiting', '等待确认执行');
            addChatMessage('system', `已生成执行计划：${data.execution_plan.total_keyframes} 个关键帧，预计 ${data.execution_plan.estimated_duration} 秒`);
        } else {
            showAlert('分析失败: ' + data.error, 'danger');
            updateTaskStatus('failed', '分析失败');
        }
    })
    .catch(error => {
        showAlert('分析出错: ' + error.message, 'danger');
        updateTaskStatus('failed', '分析失败');
        console.error('Analysis error:', error);
    })
    .finally(() => {
        document.getElementById('analyzeBtn').innerHTML = '<i class="bi bi-lightning me-2"></i>分析任务';
        updateAnalyzeButton();
    });
}

function displayDetections(detections, summary) {
    const card = document.getElementById('detectionCard');
    const summaryDiv = document.getElementById('detectionSummary');
    const listDiv = document.getElementById('detectionList');
    
    if (!detections || detections.length === 0) {
        summaryDiv.innerHTML = '<i class="bi bi-info-circle me-2"></i>未检测到物体';
        summaryDiv.className = 'alert alert-warning mb-3';
        listDiv.innerHTML = '';
        card.style.display = 'block';
        return;
    }
    
    summaryDiv.innerHTML = `<i class="bi bi-check-circle me-2"></i>${summary}`;
    summaryDiv.className = 'alert alert-info mb-3';
    
    listDiv.innerHTML = detections.map((det, index) => `
        <div class="list-group-item detection-item">
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <span class="detection-name">${det.name || '未知物体'}</span>
                    <p class="detection-desc mb-0">${det.description || ''}</p>
                </div>
                <span class="badge bg-primary rounded-pill">${(det.confidence * 100).toFixed(0)}%</span>
            </div>
            ${det.bbox ? `
            <div class="mt-2">
                <small class="text-muted">边界框: [${det.bbox.join(', ')}]</small>
                <div class="confidence-bar mt-1">
                    <div class="confidence-fill" style="width: ${det.confidence * 100}%"></div>
                </div>
            </div>
            ` : ''}
        </div>
    `).join('');
    
    card.style.display = 'block';
}

function displayExecutionPlan(plan) {
    const summaryDiv = document.getElementById('planSummary');
    const keyframesDiv = document.getElementById('keyframesList');
    const statusBadge = document.getElementById('planStatus');
    
    summaryDiv.innerHTML = `
        <div class="d-flex justify-content-between">
            <div>
                <i class="bi bi-check2-circle me-2"></i>
                <strong>${plan.summary}</strong>
            </div>
            <span class="text-muted">预计 ${plan.estimated_duration} 秒</span>
        </div>
        <div class="mt-2 small text-muted">
            任务类型: ${plan.task_type} | 置信度: ${(plan.confidence * 100).toFixed(0)}%
        </div>
    `;
    
    keyframesDiv.innerHTML = plan.keyframes.map((kf, index) => `
        <div class="keyframe-item" id="keyframe-${kf.frame_id}">
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <span class="badge bg-secondary me-2">#${kf.frame_id}</span>
                    <strong>${getKeyframeTypeLabel(kf.type)}</strong>
                </div>
                <span class="badge ${kf.gripper === 'open' ? 'bg-success' : 'bg-danger'}">
                    <i class="bi bi-hand-index${kf.gripper === 'open' ? '' : '-fill'}"></i>
                    ${kf.gripper === 'open' ? '张开' : '闭合'}
                </span>
            </div>
            <p class="text-muted mb-1 mt-1">${kf.description}</p>
            ${kf.pose ? `
            <div class="pose-info">
                位置: (${kf.pose.x || '-'}, ${kf.pose.y || '-'}, ${kf.pose.z || '-'})
                ${kf.pose.joints ? `| 关节: [${kf.pose.joints.join(', ')}]` : ''}
            </div>
            ` : ''}
            <div class="text-muted small mt-1">
                持续: ${kf.duration} 秒
            </div>
        </div>
    `).join('');
    
    statusBadge.className = 'badge bg-warning';
    statusBadge.textContent = '待确认';
}

function getKeyframeTypeLabel(type) {
    const labels = {
        'initial': '初始位置',
        'approach': '接近',
        'descend': '下降',
        'grasp': '抓取',
        'lift': '抬起',
        'move': '移动',
        'release': '放置',
        'retreat': '后退',
        'final': '完成',
        'inspect': '检查'
    };
    return labels[type] || type;
}

function executeAction(action) {
    if (!currentTaskId) {
        showAlert('没有活动的任务', 'warning');
        return;
    }
    
    const statusBadge = document.getElementById('planStatus');
    const buttons = ['confirmBtn', 'retryBtn', 'cancelBtn'];
    
    buttons.forEach(id => {
        document.getElementById(id).disabled = true;
    });
    
    if (action === 'confirm') {
        showExecutionModal();
        updateTaskStatus('executing', '正在执行...');
        addChatMessage('system', '开始执行任务...');
        
        fetch('/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                action: 'confirm',
                task_id: currentTaskId
            })
        })
        .then(response => response.json())
        .then(data => {
            hideExecutionModal();
            
            if (data.success) {
                updateKeyframesExecution(data.execution_progress);
                statusBadge.className = 'badge bg-success';
                statusBadge.textContent = '已完成';
                
                updateTaskStatus('completed', '任务完成');
                addChatMessage('system', '任务执行完成！');
                showAlert('任务执行完成！', 'success');
            } else {
                statusBadge.className = 'badge bg-danger';
                statusBadge.textContent = '失败';
                
                updateTaskStatus('failed', '执行失败');
                showAlert('执行失败: ' + data.error, 'danger');
            }
        })
        .catch(error => {
            hideExecutionModal();
            statusBadge.className = 'badge bg-danger';
            statusBadge.textContent = '失败';
            updateTaskStatus('failed', '执行失败');
            showAlert('执行出错: ' + error.message, 'danger');
            console.error('Execution error:', error);
        })
        .finally(() => {
            buttons.forEach(id => {
                document.getElementById(id).disabled = false;
            });
        });
        
    } else if (action === 'cancel') {
        fetch('/execute', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                action: 'cancel',
                task_id: currentTaskId
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                statusBadge.className = 'badge bg-secondary';
                statusBadge.textContent = '已取消';
                
                updateTaskStatus('cancelled', '任务已取消');
                addChatMessage('system', '任务已取消');
                showAlert('任务已取消', 'info');
            }
        });
        
        buttons.forEach(id => {
            document.getElementById(id).disabled = false;
        });
        
    } else if (action === 'retry') {
        statusBadge.className = 'badge bg-info';
        statusBadge.textContent = '重新规划中...';
        
        updateTaskStatus('planning', '重新规划中...');
        addChatMessage('system', '重新规划任务...');
        
        analyzeTask();
    }
}

function updateKeyframesExecution(progress) {
    progress.forEach(item => {
        const element = document.getElementById(`keyframe-${item.frame_id}`);
        if (element) {
            element.classList.add('executed');
            const badge = document.createElement('span');
            badge.className = 'badge bg-success ms-2';
            badge.innerHTML = '<i class="bi bi-check"></i> 已执行';
            element.querySelector('strong').after(badge);
        }
    });
}

function showExecutionModal() {
    const modal = new bootstrap.Modal(document.getElementById('executionModal'));
    modal.show();
    
    if (currentExecutionPlan) {
        const total = currentExecutionPlan.keyframes.length;
        let current = 0;
        
        const interval = setInterval(() => {
            current++;
            const progress = Math.min((current / total) * 100, 95);
            document.getElementById('executionProgress').style.width = progress + '%';
            
            if (current < total) {
                document.getElementById('executionText').textContent = 
                    `执行关键帧 ${current}/${total}: ${currentExecutionPlan.keyframes[current-1].description}`;
            } else {
                clearInterval(interval);
            }
        }, 500);
    }
}

function hideExecutionModal() {
    document.getElementById('executionProgress').style.width = '100%';
    document.getElementById('executionText').textContent = '执行完成！';
    
    setTimeout(() => {
        const modal = bootstrap.Modal.getInstance(document.getElementById('executionModal'));
        if (modal) modal.hide();
    }, 1000);
}

function exportTask() {
    if (!currentTaskId) {
        showAlert('没有可导出的任务', 'warning');
        return;
    }
    
    const btn = event.target;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>导出中...';
    btn.disabled = true;
    
    fetch('/export', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            task_id: currentTaskId,
            format: 'json'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            addChatMessage('system', `任务记录已导出: ${data.export_path}`);
            showAlert('导出成功！正在下载...', 'success');
            
            window.location.href = data.download_url;
        } else {
            showAlert('导出失败: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        showAlert('导出出错: ' + error.message, 'danger');
        console.error('Export error:', error);
    })
    .finally(() => {
        btn.innerHTML = originalText;
        btn.disabled = false;
    });
}

function addChatMessage(type, content) {
    const chatHistory = document.getElementById('chatHistory');
    
    if (chatMessages.length === 0) {
        chatHistory.innerHTML = '';
    }
    
    const now = new Date();
    const timestamp = now.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    messageDiv.innerHTML = `
        <div>${content}</div>
        <div class="timestamp">${timestamp}</div>
    `;
    
    chatHistory.appendChild(messageDiv);
    chatMessages.push({ type, content, timestamp });
    
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function updateTaskStatus(status, text) {
    const statusElement = document.getElementById('taskStatus');
    const statusClass = `status-${status}`;
    
    statusElement.innerHTML = `<span class="status-indicator ${statusClass}"></span>${text}`;
}

function showAlert(message, type) {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    setTimeout(() => {
        const alert = bootstrap.Alert.getInstance(alertDiv);
        if (alert) alert.close();
    }, 5000);
}
