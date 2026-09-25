/**
 * Loan Default Prediction System — Frontend JavaScript
 * Single model: Gradient Boosting Classifier
 * All predictions via POST /api/predict (no client-side ML).
 */
document.addEventListener('DOMContentLoaded', () => {

    // ---- Mobile nav ----
    const menuToggle = document.querySelector('.menu-toggle');
    const navLinks = document.querySelector('.nav-links');
    if (menuToggle && navLinks) {
        menuToggle.addEventListener('click', () => navLinks.classList.toggle('open'));
    }

    // ---- Toast ----
    function showToast(msg, type, duration) {
        type = type || 'info'; duration = duration || 3000;
        let c = document.getElementById('toast-container');
        if (!c) { c = document.createElement('div'); c.className = 'toast-container'; c.id = 'toast-container'; document.body.appendChild(c); }
        const t = document.createElement('div');
        t.className = 'toast toast-' + type;
        t.textContent = msg;
        c.appendChild(t);
        setTimeout(() => { t.style.opacity = '0'; setTimeout(() => t.remove(), 250); }, duration);
    }

    // ---- Loading ----
    function showLoading() { const o = document.getElementById('loading-overlay'); if (o) o.classList.remove('hidden'); }
    function hideLoading() { const o = document.getElementById('loading-overlay'); if (o) o.classList.add('hidden'); }

    // ---- Utility ----
    function fmtPct(v) {
        if (v == null || isNaN(v)) return 'N/A';
        var n = parseFloat(v);
        return n <= 1 ? (n * 100).toFixed(2) + '%' : n.toFixed(2) + '%';
    }

    // ---- HOME PAGE ----
    const statAcc = document.getElementById('stat-accuracy');
    if (statAcc) {
        fetch('/api/health').then(r => r.json()).then(d => {
            if (d.accuracy != null) statAcc.textContent = fmtPct(d.accuracy);
        }).catch(() => {});
    }

    // ---- PREDICTION PAGE ----
    const form = document.getElementById('predict-form');
    if (form) {
        // Load presets
        let presets = {};
        fetch('/api/default-applicant').then(r => r.json()).then(d => { presets = d; }).catch(() => {});

        document.querySelectorAll('.preset-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const key = btn.dataset.preset;
                if (presets[key]) {
                    fillForm(presets[key]);
                    showToast('Loaded sample data', 'info');
                    document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
                    btn.classList.add('active');
                }
            });
        });

        function fillForm(d) {
            if (!d) return;
            ['age','income','loanAmount','creditScore','monthsEmployed','numCreditLines','interestRate','dtiRatio'].forEach(k => {
                const el = document.getElementById(k); if (el && d[k] !== undefined) el.value = d[k];
            });
            ['education','employmentType','maritalStatus','loanTerm','loanPurpose'].forEach(k => {
                const el = document.getElementById(k); if (el && d[k] !== undefined) el.value = String(d[k]);
            });
            ['hasMortgage','hasDependents','hasCoSigner'].forEach(k => {
                const el = document.getElementById(k); if (el && d[k] !== undefined) el.checked = !!d[k];
            });
            document.querySelectorAll('.form-group.error').forEach(g => g.classList.remove('error'));
        }

        function validate() {
            let ok = true;
            document.querySelectorAll('.form-group.error').forEach(g => g.classList.remove('error'));
            [
                {id:'age',min:18,max:69,int:true}, {id:'income',min:15000,max:150000},
                {id:'loanAmount',min:5000,max:250000}, {id:'creditScore',min:300,max:850,int:true},
                {id:'monthsEmployed',min:0,max:120,int:true}, {id:'numCreditLines',min:1,max:4,int:true},
                {id:'interestRate',min:2,max:25}, {id:'dtiRatio',min:0.1,max:0.9}
            ].forEach(r => {
                const el = document.getElementById(r.id); if (!el) return;
                const v = parseFloat(el.value);
                const g = el.closest('.form-group');
                if (el.value.trim()==='' || isNaN(v) || v<r.min || v>r.max || (r.int && !Number.isInteger(v))) {
                    if (g) g.classList.add('error'); ok = false;
                }
            });
            return ok;
        }

        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!validate()) { showToast('Please correct the form errors.', 'error'); return; }

            const payload = {
                age: parseInt(document.getElementById('age').value),
                income: parseFloat(document.getElementById('income').value),
                loanAmount: parseFloat(document.getElementById('loanAmount').value),
                creditScore: parseInt(document.getElementById('creditScore').value),
                monthsEmployed: parseInt(document.getElementById('monthsEmployed').value),
                numCreditLines: parseInt(document.getElementById('numCreditLines').value),
                interestRate: parseFloat(document.getElementById('interestRate').value),
                loanTerm: parseInt(document.getElementById('loanTerm').value),
                dtiRatio: parseFloat(document.getElementById('dtiRatio').value),
                education: document.getElementById('education').value,
                employmentType: document.getElementById('employmentType').value,
                maritalStatus: document.getElementById('maritalStatus').value,
                loanPurpose: document.getElementById('loanPurpose').value,
                hasMortgage: document.getElementById('hasMortgage').checked,
                hasDependents: document.getElementById('hasDependents').checked,
                hasCoSigner: document.getElementById('hasCoSigner').checked
            };

            showLoading();
            try {
                const res = await fetch('/api/predict', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const data = await res.json();
                if (!res.ok || data.status !== 'success') throw new Error(data.message || 'Server error');
                showResult(data, payload);
                showToast('Prediction complete.', 'success');
            } catch (err) {
                showToast('Error: ' + err.message, 'error');
            } finally { hideLoading(); }
        });

        function showResult(data, sent) {
            const sec = document.getElementById('result-section');
            if (!sec) return;
            sec.classList.remove('hidden');
            setTimeout(() => sec.scrollIntoView({behavior:'smooth',block:'start'}), 80);

            const isNo = data.prediction_text === 'No';
            const banner = document.getElementById('result-banner');
            banner.className = 'result-banner ' + (isNo ? 'success' : 'danger');
            document.getElementById('result-icon').textContent = isNo ? '\u2714' : '\u2718';
            document.getElementById('result-icon').style.color = isNo ? 'var(--success)' : 'var(--danger)';
            document.getElementById('result-title').textContent = 'Default Prediction: ' + data.prediction_text;
            document.getElementById('result-subtitle').textContent =
                isNo ? 'The model predicts this applicant will not default.' : 'The model predicts this applicant is at risk of default.';

            const pred = document.getElementById('result-prediction');
            pred.textContent = data.prediction_text;
            pred.style.color = isNo ? 'var(--success)' : 'var(--danger)';

            document.getElementById('result-probability').textContent = data.default_probability.toFixed(2) + '%';
            const bar = document.getElementById('result-probability-bar');
            bar.style.width = Math.min(data.default_probability, 100) + '%';
            bar.style.backgroundColor = data.default_probability > 50 ? 'var(--danger)' : 'var(--success)';

            document.getElementById('result-risk-tier').textContent = data.risk_tier;

            // Factors
            const pos = document.getElementById('positive-factors');
            pos.innerHTML = '';
            (data.positive_factors || []).forEach(f => { const li = document.createElement('li'); li.className='pos'; li.textContent=f; pos.appendChild(li); });
            if (!data.positive_factors || !data.positive_factors.length) { const li = document.createElement('li'); li.textContent='None identified'; li.style.color='var(--text-muted)'; pos.appendChild(li); }

            const neg = document.getElementById('risk-factors');
            neg.innerHTML = '';
            (data.risk_factors || []).forEach(f => { const li = document.createElement('li'); li.className='neg'; li.textContent=f; neg.appendChild(li); });
            if (!data.risk_factors || !data.risk_factors.length) { const li = document.createElement('li'); li.textContent='None identified'; li.style.color='var(--text-muted)'; neg.appendChild(li); }

            // Summary table
            const summ = document.getElementById('applicant-summary');
            if (summ && sent) {
                const labels = {
                    age:'Age', income:'Annual Income', loanAmount:'Loan Amount',
                    creditScore:'Credit Score', monthsEmployed:'Months Employed',
                    numCreditLines:'Credit Lines', interestRate:'Interest Rate',
                    loanTerm:'Loan Term', dtiRatio:'DTI Ratio',
                    education:'Education', employmentType:'Employment Type',
                    maritalStatus:'Marital Status', loanPurpose:'Loan Purpose',
                    hasMortgage:'Has Mortgage', hasDependents:'Has Dependents', hasCoSigner:'Has Co-Signer'
                };
                const fmt = (k,v) => {
                    if (typeof v === 'boolean') return v ? 'Yes' : 'No';
                    if (k==='income'||k==='loanAmount') return '$'+Number(v).toLocaleString();
                    if (k==='interestRate') return v+'%';
                    if (k==='loanTerm') return v+' months';
                    return v;
                };
                let rows = '';
                for (const [k,lbl] of Object.entries(labels)) {
                    if (sent[k]!==undefined) rows+='<tr><th>'+lbl+'</th><td>'+fmt(k,sent[k])+'</td></tr>';
                }
                summ.innerHTML = '<table class="summary-table">'+rows+'</table>';
            }
        }

        const resetBtn = document.getElementById('reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                document.getElementById('result-section').classList.add('hidden');
                form.scrollIntoView({behavior:'smooth'});
            });
        }
    }

    // ---- MODELS PAGE ----
    const metricsTable = document.getElementById('metrics-table');
    if (metricsTable) {
        fetch('/api/models').then(r => r.json()).then(data => {
            // Description
            const descEl = document.getElementById('model-description');
            if (descEl) descEl.textContent = data.description || '';

            // Metrics table
            const tbody = metricsTable.querySelector('tbody');
            tbody.innerHTML = '';
            const metrics = [
                ['Accuracy', data.accuracy],
                ['Precision', data.precision],
                ['Recall', data.recall],
                ['F1-Score', data.f1_score],
                ['ROC-AUC', data.roc_auc],
                ['CV Mean Accuracy', data.cv_mean_accuracy]
            ];
            metrics.forEach(([label, val]) => {
                const tr = document.createElement('tr');
                tr.innerHTML = '<td><strong>'+label+'</strong></td><td>'+fmtPct(val)+'</td>';
                tbody.appendChild(tr);
            });

            // Confusion matrix
            const cm = data.confusion_matrix;
            if (cm) {
                const cmCard = document.getElementById('cm-card');
                const cmTable = document.getElementById('cm-table');
                if (cmCard && cmTable) {
                    cmCard.style.display = 'block';
                    cmTable.innerHTML =
                        '<tr><th></th><th>Predicted Negative</th><th>Predicted Positive</th></tr>'+
                        '<tr><th>Actual Negative</th><td>'+cm.tn.toLocaleString()+'</td><td>'+cm.fp.toLocaleString()+'</td></tr>'+
                        '<tr><th>Actual Positive</th><td>'+cm.fn.toLocaleString()+'</td><td>'+cm.tp.toLocaleString()+'</td></tr>';
                }
            }
        }).catch(err => {
            metricsTable.querySelector('tbody').innerHTML = '<tr><td colspan="2" style="color:var(--danger)">Failed to load. Is Flask running?</td></tr>';
        });
    }
});
