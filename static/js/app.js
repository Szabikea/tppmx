/**
 * Tippmix AI Segéd - Frontend JavaScript
 * =======================================
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('🎯 Tippmix AI Segéd loaded');

    // API státusz frissítés
    updateAPIStatus();

    // Tooltips inicializálása (Bootstrap)
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
});

/**
 * API státusz lekérése és frissítése
 */
async function updateAPIStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        const usageEl = document.getElementById('api-usage');
        if (usageEl) {
            usageEl.textContent = `${data.daily_usage}/${data.daily_limit}`;

            // Szín változtatás usage alapján
            const percent = (data.daily_usage / data.daily_limit) * 100;
            const badge = usageEl.closest('.api-usage-badge');

            if (percent > 80) {
                badge.style.borderColor = 'var(--accent-red)';
                badge.style.color = 'var(--accent-red)';
            } else if (percent > 50) {
                badge.style.borderColor = 'var(--accent-yellow)';
                badge.style.color = 'var(--accent-yellow)';
            }
        }
    } catch (error) {
        console.error('API status fetch failed:', error);
    }
}

/**
 * Gyors meccs elemzés AJAX betöltése
 */
async function loadQuickAnalysis(fixtureId, targetElement) {
    try {
        targetElement.innerHTML = '<div class="text-center py-4"><div class="spinner-border text-primary" role="status"></div></div>';

        const response = await fetch(`/api/quick-analysis/${fixtureId}`);
        const data = await response.json();

        if (data.error) {
            targetElement.innerHTML = `<div class="alert alert-warning">${data.error}</div>`;
            return;
        }

        // Render tips
        let html = '<div class="row g-2">';
        data.tips.forEach(tip => {
            html += `
                <div class="col-md-6">
                    <div class="tip-card ${tip.confidence >= 4 ? 'tip-high-confidence' : ''}">
                        <div class="tip-header">
                            <span class="tip-type">${tip.description}</span>
                            <span class="tip-confidence">${'★'.repeat(tip.confidence)}${'☆'.repeat(5 - tip.confidence)}</span>
                        </div>
                        <div class="tip-body">
                            <div class="tip-probability">
                                <div class="probability-bar">
                                    <div class="probability-fill" style="width: ${tip.probability}%"></div>
                                </div>
                                <span class="probability-value">${tip.probability}%</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        html += '</div>';

        targetElement.innerHTML = html;

    } catch (error) {
        console.error('Quick analysis failed:', error);
        targetElement.innerHTML = '<div class="alert alert-danger">Hiba az elemzés betöltésekor</div>';
    }
}

/**
 * Cache frissítése
 */
async function refreshCache() {
    try {
        const response = await fetch('/api/refresh-cache', { method: 'POST' });
        const data = await response.json();

        if (data.success) {
            showToast('success', data.message);
        }
    } catch (error) {
        showToast('error', 'Hiba a cache frissítésekor');
    }
}

/**
 * Toast üzenet megjelenítése
 */
function showToast(type, message) {
    // Egyszerű alert, de lehetne szebb toast komponens is
    const icon = type === 'success' ? '✅' : '❌';
    alert(`${icon} ${message}`);
}

/**
 * Probability bar animáció
 */
function animateProbabilityBars() {
    const bars = document.querySelectorAll('.probability-fill');
    bars.forEach(bar => {
        const width = bar.style.width;
        bar.style.width = '0';
        setTimeout(() => {
            bar.style.width = width;
        }, 100);
    });
}

// Auto-refresh API status every 30 seconds
setInterval(updateAPIStatus, 30000);
