document.addEventListener('DOMContentLoaded', () => {
    const copyBtn = document.getElementById('copy-btn');
    const bibtexCode = document.getElementById('bibtex-code');

    // BibTeX 复制功能
    if (copyBtn) {
        copyBtn.addEventListener('click', () => {
            const text = bibtexCode.innerText;
            
            navigator.clipboard.writeText(text).then(() => {
                // 修改按钮状态
                const originalText = copyBtn.innerText;
                copyBtn.innerText = '已复制！';
                copyBtn.style.background = '#e6fffa';
                copyBtn.style.borderColor = '#38b2ac';

                // 2秒后恢复
                setTimeout(() => {
                    copyBtn.innerText = originalText;
                    copyBtn.style.background = '#fff';
                    copyBtn.style.borderColor = '#ccc';
                }, 2000);
            }).catch(err => {
                console.error('复制失败: ', err);
            });
        });
    }

    // 这里可以添加更多交互逻辑，例如图片滚动加载、平滑滚动等
    console.log("SEER Project Page Initialized.");
});
