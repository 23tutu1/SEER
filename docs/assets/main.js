/**
 * 复制 BibTeX 到剪贴板
 */
function copyBib() {
    const bibText = document.querySelector('pre code').innerText;
    navigator.clipboard.writeText(bibText).then(() => {
        const btn = document.querySelector('.copy-button');
        btn.innerText = 'Copied!';
        btn.style.backgroundColor = '#e8f5e9';
        
        setTimeout(() => {
            btn.innerText = 'Copy';
            btn.style.backgroundColor = '#fff';
        }, 2000);
    });
}

console.log("SEER Project Page (RenderFormer Style) Loaded.");
