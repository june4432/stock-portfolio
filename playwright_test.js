const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();

  try {
    // Navigate to the application
    await page.goto('http://localhost:9010');

    // Click the '거래관리' tab
    await page.click('button[data-tab="trading"]');

    // Wait for the transaction list to be visible (assuming it has an ID 'all-transaction-list')
    await page.waitForSelector('#all-transaction-list', { state: 'visible' });

    // Take a screenshot of the entire page
    await page.screenshot({ path: 'transaction_management_ui.png', fullPage: true });

    console.log('Screenshot taken: transaction_management_ui.png');

  } catch (error) {
    console.error('Playwright test failed:', error);
  } finally {
    await browser.close();
  }
})();