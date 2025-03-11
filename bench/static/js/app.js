class BenchmarkUI {
    constructor() {
        this.socket = null;
        this.retries = 0;
        this.maxRetries = 5;
        this.reconnectTimeout = null;

        this.initializeElements();
        this.initializeEventListeners();
        this.loadDatasets();
    }

    initializeElements() {
        this.elements = {
            uploadForm: document.getElementById('upload-form'),
            datasetsList: document.getElementById('datasets-list'),
            datasetSelect: document.getElementById('dataset-select'),
            benchmarkForm: document.getElementById('benchmark-form'),
            outputConsole: document.getElementById('output-console'),
            visualizationContainer: document.getElementById('visualization-container'),
            cleanResultsBtn: document.getElementById('clean-results-btn')
        };
    }

    initializeEventListeners() {
        this.elements.uploadForm.addEventListener('submit', (e) => this.handleUpload(e));
        this.elements.benchmarkForm.addEventListener('submit', (e) => this.runBenchmark(e));
        this.elements.cleanResultsBtn.addEventListener('click', (e) => this.handleCleanResults(e));

    }

    async handleUpload(e) {
        e.preventDefault();
        const fileInput = this.elements.uploadForm.querySelector('input[type="file"]');
        const statusElement = document.getElementById('upload-status');

        if (!fileInput.files[0]) {
            this.showStatus('파일을 선택해주세요.', 'error', statusElement);
            return;
        }

        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        this.showStatus('파일 업로드 중...', 'info', statusElement);

        try {
            const response = await fetch('/upload/', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.showStatus(`"${data.filename}" 업로드 성공`, 'success', statusElement);
                fileInput.value = '';
                this.loadDatasets();
            } else {
                this.showStatus(`업로드 실패: ${data.detail}`, 'error', statusElement);
            }
        } catch (err) {
            this.showStatus(`업로드 오류: ${err.message}`, 'error', statusElement);
        }
    }

    showStatus(message, type, element) {
        element.textContent = message;
        element.className = `alert ${type}`;
        element.style.display = 'block';
    }

    async loadDatasets() {
        try {
            const response = await fetch('/datasets/');
            const data = await response.json();

            if (response.ok) {
                this.updateDatasetList(data.datasets);
                this.updateDatasetSelect(data.datasets);
            } else {
                this.elements.datasetsList.innerHTML = '<p>데이터셋 불러오기 실패</p>';
            }
        } catch (err) {
            this.elements.datasetsList.innerHTML = `<p>오류: ${err.message}</p>`;
        }
    }

    updateDatasetList(datasets) {
        const listElement = this.elements.datasetsList;
        listElement.innerHTML = datasets.length > 0
            ? `<ul>${datasets.map(d => `<li>${d}</li>`).join('')}</ul>`
            : '<p>업로드된 데이터셋 없음</p>';
    }

    updateDatasetSelect(datasets) {
        const select = this.elements.datasetSelect;
        select.innerHTML = '<option value="">선택해주세요</option>';
        datasets.forEach(dataset => {
            const option = document.createElement('option');
            option.value = dataset;
            option.textContent = dataset;
            select.appendChild(option);
        });
    }

    async runBenchmark(e) {
        e.preventDefault();
        const formData = new FormData(this.elements.benchmarkForm);
        const requestData = Object.fromEntries(formData.entries());

        if (!this.validateInputs(requestData)) return;

        this.clearOutput();
        this.showBenchmarkStatus('벤치마크 시작 중...');

        try {
            const response = await fetch('/run-benchmark/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    base_url: requestData.base_url,
                    dataset_name: requestData.dataset_name,
                    dataset_type: requestData.dataset_type,
                    model_name: requestData.model_name,
                    num_prompts: parseInt(requestData.num_prompts)
                })
            });

            const data = await response.json();
            if (response.ok) {
                this.connectWebSocket();
            } else {
                this.appendOutput(`오류: ${data.detail}`);
            }
        } catch (err) {
            this.appendOutput(`요청 실패: ${err.message}`);
        }
    }

    validateInputs(data) {
        if (!data.dataset_name) {
            alert('데이터셋 파일을 선택해주세요.');
            return false;
        }
        return true;
    }

    clearOutput() {
        this.elements.outputConsole.textContent = '';
    }

    showBenchmarkStatus(message) {
        this.elements.visualizationContainer.innerHTML = `<p>${message}</p>`;
    }

    connectWebSocket() {
        if (this.socket) {
            this.socket.close();
        }

        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.socket = new WebSocket(`${protocol}//${window.location.host}/ws`);

        this.socket.onmessage = (event) => {
            if (event.data === 'BENCHMARK_COMPLETED') {
                this.loadVisualizations();
            } else {
                this.appendOutput(event.data);
            }
        };

        this.socket.onclose = () => {
            if (this.retries < this.maxRetries) {
                this.retries++;
                this.reconnectTimeout = setTimeout(
                    () => this.connectWebSocket(),
                    Math.pow(2, this.retries) * 1000
                );
            }
        };

        this.socket.onerror = (error) => {
            console.error('WebSocket Error:', error);
        };
    }

    appendOutput(text) {
        this.elements.outputConsole.textContent += text + '\n';
        this.elements.outputConsole.scrollTop = this.elements.outputConsole.scrollHeight;
    }

    async loadVisualizations() {
        try {
            const response = await fetch('/visualizations');
            const data = await response.json();

            if (data.visualizations.length > 0) {
                this.elements.visualizationContainer.innerHTML = data.visualizations
                    .map(img => `<img src="/visualization/${img}" class="visualization-image">`)
                    .join('');
            } else {
                this.showBenchmarkStatus('시각화 결과 없음');
            }
        } catch (error) {
            this.showBenchmarkStatus('결과 불러오기 실패');
        }
    }

    async handleCleanResults(e) {
        try {
            const response = await fetch('/clean-results/', { method: 'DELETE' });
            const data = await response.json();
            ui.showStatus(data.status, data.status, statusElement);
            ui.loadVisualizations();
        } catch (error) {
            ui.showStatus(`Failed to clean: ${error.message}`, 'error');
        }
    }

}

document.addEventListener('DOMContentLoaded', () => {
    new BenchmarkUI();
});
