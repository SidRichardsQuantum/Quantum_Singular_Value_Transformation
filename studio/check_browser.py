"""Optional end-to-end smoke check against a running, disposable studio server.

Install playwright and its Chromium browser separately. This check creates
real experiment records; always launch the server with a temporary data-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect, sync_playwright


def check_image(page, selector):
    images = page.locator(selector)
    assert images.count() > 0
    for index in range(images.count()):
        image = images.nth(index)
        expect(image).to_have_js_property("complete", True, timeout=30000)
        assert image.evaluate("element => element.naturalWidth") > 0


def check_history_browser(browser, url):
    """Exercise paging and polling with deterministic saved-record fixtures."""
    page = browser.new_page()
    page.clock.install()
    catalogue = page.request.get(f"{url}/api/catalogue").json()
    preset = next(p for p in catalogue["presets"] if p["id"] == "sign-cookbook")
    records = [
        {
            "id": f"{index:032x}",
            "request": {
                "workflow": preset["workflow"],
                "settings": preset["settings"],
                "schema_version": catalogue["schema_version"],
            },
            "status": "completed",
            "created_at": "2026-09-22T00:00:00+00:00",
            "favorite": False,
            "artifacts": [],
            "metrics": {},
        }
        for index in range(27)
    ]
    calls = []
    mode = {"active": 0, "failed": False}

    def history(route):
        query = parse_qs(urlsplit(route.request.url).query)
        offset = int(query.get("offset", ["0"])[0])
        calls.append(offset)
        if mode["failed"]:
            route.fulfill(status=503, json={"error": "Test connection failure"})
            return
        route.fulfill(
            json={
                "runs": records[offset : offset + 24],
                "total": 27,
                "matching": 27,
                "active": mode["active"],
                "offset": offset,
                "limit": 24,
            }
        )

    page.route("**/api/runs?*", history)
    page.goto(url)
    expect(page.locator(".card")).to_have_count(24)
    page.locator(".card").first.get_by_role(
        "button", name="Compare", exact=True
    ).click()
    page.click("#next-page")
    expect(page.locator(".card")).to_have_count(3)
    expect(page.locator("#page-status")).to_have_text("25–27 of 27")
    expect(page.locator("#next-page")).to_be_disabled()
    page.locator(".card").first.get_by_role(
        "button", name="Compare", exact=True
    ).click()
    expect(page.locator("#selection-count")).to_contain_text("2 selected")
    page.click("#previous-page")
    expect(page.locator(".card")).to_have_count(24)
    expect(
        page.locator(".card").first.get_by_role("button", name="✓ Selected")
    ).to_be_visible()
    assert calls[:3] == [0, 24, 0]

    # An active report can change without losing expanded details or focus.
    records[0]["status"] = "executing"
    records[0]["events"] = []
    page.route(
        f"**/api/runs/{records[0]['id']}",
        lambda route: route.fulfill(json={**records[0], "report": {}}),
    )
    page.locator(".card").first.get_by_role("button", name="View", exact=True).click()
    # Clicking View starts an asynchronous fetch; wait for the rendered report.
    expect(page.locator("#report-reproducibility")).to_be_visible()
    page.evaluate("""() => {
      const section = document.querySelector('#report-reproducibility');
      const summary = section.querySelector('details:last-of-type summary');
      summary.click(); summary.focus();
      window.savedScroll = document.querySelector('#viewer').scrollTop;
      window.phaseSection = document.querySelector('#report-phases');
    }""")
    records[0]["events"] = [{"status": "saving_artifacts"}]
    records[0]["status"] = "saving_artifacts"
    page.evaluate("refresh()")
    assert page.evaluate("""() => {
      const section = document.querySelector('#report-reproducibility');
      const details = section.querySelector('details:last-of-type');
      const summary = details.querySelector('summary');
      return details.open && document.activeElement === summary;
    }""")
    assert page.evaluate("document.querySelector('#viewer').scrollTop === savedScroll")
    assert page.evaluate("document.querySelector('#report-phases') === phaseSection")
    page.click("#close-viewer")

    # Idle refresh discovers activity; subsequent refreshes use two seconds.
    mode["active"] = 1
    with page.expect_response("**/api/runs?*"):
        page.clock.fast_forward(15000)
    page.wait_for_function("state.active === 1")
    with page.expect_response("**/api/runs?*"):
        page.clock.fast_forward(2000)
    page.evaluate("""() => {
      Object.defineProperty(document, 'hidden', {configurable: true, value: true});
      document.dispatchEvent(new Event('visibilitychange'));
    }""")
    before = len(calls)
    page.clock.fast_forward(60000)
    assert len(calls) == before
    with page.expect_response("**/api/runs?*"):
        page.evaluate("""() => {
          Object.defineProperty(document, 'hidden', {configurable: true, value: false});
          document.dispatchEvent(new Event('visibilitychange'));
        }""")
    mode["failed"] = True
    with page.expect_response("**/api/runs?*"):
        page.clock.fast_forward(2000)
    expect(page.locator("#notice")).to_contain_text("Test connection failure")
    before = len(calls)
    page.clock.fast_forward(29000)
    assert len(calls) == before
    mode["failed"] = False
    with page.expect_response("**/api/runs?*"):
        page.clock.fast_forward(1000)
    expect(page.locator("#notice")).to_be_empty()
    page.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--screenshot", type=Path, default=Path("/tmp/qsvt-studio.png"))
    args = parser.parse_args()
    errors = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        expect(page.locator("#workflow option")).to_have_count(7)
        expect(page.locator("#workflow")).to_have_value("sign")
        expect(page.locator("#preset")).to_have_value("sign-cookbook")
        expect(page.locator("#setting-attempt_synthesis")).not_to_be_checked()
        page.click("#package-defaults")
        expect(page.locator("#setting-attempt_synthesis")).to_be_checked()
        page.select_option("#preset", "sign-cookbook")
        expect(page.locator("#setting-degree")).to_have_attribute("step", "2")
        expect(page.locator("#setting-degree")).to_have_value("13")
        # Retain distinct drafts, including unfinished input, across reloads.
        page.fill("#setting-degree", "")
        page.select_option("#workflow", "inverse")
        page.fill("#setting-degree", "15")
        page.select_option("#workflow", "sign")
        expect(page.locator("#setting-degree")).to_have_value("")
        page.reload()
        expect(page.locator("#setting-degree")).to_have_value("")
        page.select_option("#workflow", "inverse")
        expect(page.locator("#setting-degree")).to_have_value("15")
        page.select_option("#preset", "sign-cookbook")
        page.click("#run")
        expect(page.locator(".card .status.completed")).to_have_count(1, timeout=60000)
        card = page.locator(".card").first
        card.get_by_role("button", name="☆ Save", exact=True).click()
        expect(card.get_by_role("button", name="★ Saved", exact=True)).to_be_visible()
        card.get_by_role("button", name="View", exact=True).click()
        expect(page.locator("#viewer")).to_be_visible()
        expect(page.locator("#viewer-body")).to_contain_text("diagnostics.max_error")
        expect(page.locator("#report-overview")).to_contain_text(
            "Execution: Not requested"
        )
        expect(page.locator("#report-overview")).to_contain_text(
            "No acceptance verdict recorded"
        )
        expect(page.locator(".report-nav a")).to_have_count(5)
        # Metadata-only history changes must leave the reader's DOM and focus intact.
        page.evaluate("""() => {
          const section = document.querySelector('#report-reproducibility');
          const summary = section.querySelector('details:last-of-type summary');
          summary.click(); summary.focus();
          window.readerNode = summary;
          window.readerScroll = document.querySelector('#viewer').scrollTop;
        }""")
        page.evaluate("""async () => {
          await api(`/api/runs/${state.viewing}/favorite`, {favorite: false});
          await refresh();
          await api(`/api/runs/${state.viewing}/favorite`, {favorite: true});
          await refresh();
        }""")
        assert page.evaluate("document.activeElement === window.readerNode")
        assert page.evaluate("window.readerNode.parentElement.open")
        assert page.evaluate(
            "document.querySelector('#viewer').scrollTop === window.readerScroll"
        )
        with page.expect_download() as downloaded:
            page.get_by_role(
                "button", name="Export configuration", exact=True
            ).last.click()
        request = json.loads(Path(downloaded.value.path()).read_text())
        page.get_by_role("button", name="Reuse configuration", exact=True).click()
        expect(page.locator("#setting-degree")).to_have_value("13")
        assert page.locator("#setting-attempt_synthesis").is_checked() is False
        assert request["settings"]["degree"] == 13
        page.fill("#setting-degree", "9")
        page.click("#run")
        expect(page.locator(".card .status.completed")).to_have_count(2, timeout=60000)
        for card in page.locator(".card").all():
            card.get_by_role("button", name="Compare", exact=True).click()
        page.click("#compare")
        expect(page.locator("#comparison")).to_be_visible()
        expect(page.locator("#comparison-body")).to_contain_text(
            "diagnostics.max_error"
        )
        expect(page.locator("#comparison-body .large-plot")).to_be_visible()
        check_image(page, "#comparison-body img")
        page.click("#close-comparison")
        page.click("#clear-selection")
        # A third run with another target must not compare to the first two.
        page.fill("#setting-gamma", "0.4")
        page.click("#run")
        expect(page.locator(".card .status.completed")).to_have_count(3, timeout=60000)
        for card in page.locator(".card").all()[:2]:
            card.get_by_role("button", name="Compare", exact=True).click()
        page.click("#compare")
        expect(page.locator("#notice")).to_contain_text("Incompatible")
        page.reload()
        page.check("#favorites")
        expect(page.locator(".card")).to_have_count(1)
        page.reload()
        expect(page.locator("#favorites")).to_be_checked()
        expect(page.locator(".card")).to_have_count(1)
        expect(page.locator(".card")).to_contain_text("★ Saved")
        page.uncheck("#favorites")
        page.fill("#search", "no-such-experiment")
        expect(page.locator(".empty")).to_contain_text("No matching")
        page.fill("#search", "")
        expect(page.locator(".card")).to_have_count(3)
        page.select_option("#preset", label="Hamiltonian · six-site coherent QNode")
        expect(page.locator("#setting-time")).to_have_value("1.4")
        expect(page.locator("#setting-execute_qsvt")).to_be_checked()
        page.click("#run")
        expect(page.locator(".card .status.completed")).to_have_count(4, timeout=60000)
        page.select_option("#workflow-filter", "hamiltonian_simulation")
        expect(page.locator(".card")).to_have_count(1)
        expect(page.locator(".card")).to_contain_text("Analytic QNode requested")
        page.locator(".card").get_by_role("button", name="View", exact=True).click()
        expect(page.locator("#viewer-body")).to_contain_text(
            "full_qsvt_acceptance: true"
        )
        check_image(page, "#viewer-body img")
        page.get_by_role("button", name="Reuse configuration", exact=True).click()
        expect(page.locator("#setting-execute_qsvt")).to_be_checked()
        expect(page.locator("#setting-degree")).to_have_value("12")
        page.uncheck("#setting-execute_qsvt")
        page.fill("#setting-degree", "8")
        page.click("#run")
        expect(page.locator(".card .status.completed")).to_have_count(2, timeout=60000)
        page.select_option("#execution-filter", "false")
        expect(page.locator(".card")).to_have_count(1)
        expect(page.locator(".card")).to_contain_text("no QNode or phase synthesis")
        page.select_option("#execution-filter", "true")
        expect(page.locator(".card")).to_have_count(1)
        expect(page.locator(".card")).to_contain_text("Analytic QNode requested")
        page.select_option("#execution-filter", "")
        expect(page.locator(".card")).to_have_count(2)
        for card in page.locator(".card").all():
            card.get_by_role("button", name="Compare", exact=True).click()
        page.click("#compare")
        expect(page.locator("#comparison-body")).to_contain_text("state_relative_error")
        check_image(page, "#comparison-body img")
        page.click("#close-comparison")
        # Restore an old request without changing its scientific settings.
        legacy = {
            **request,
            "schema_version": "1.0",
            "settings": dict(request["settings"]),
        }
        for key in (
            "angle_solver",
            "phase_reconstruction_tolerance",
            "reconstruction_num_points",
        ):
            legacy["settings"].pop(key)
        page.locator("#import").set_input_files(
            {
                "name": "legacy.json",
                "mimeType": "application/json",
                "buffer": json.dumps(legacy).encode(),
            }
        )
        expect(page.locator("#notice")).to_contain_text("upgraded to 1.2")
        expect(page.locator("#setting-degree")).to_have_value("13")
        expect(page.locator("#setting-angle_solver")).to_have_value('"root-finding"')
        page.select_option("#workflow-filter", "sign")
        page.select_option("#preset", "sign-phases")
        page.locator("#advanced-settings summary").click()
        expect(page.locator("#setting-angle_solver")).to_have_value('"iterative"')
        page.click("#run")
        expect(page.locator(".card .status.completed")).to_have_count(4, timeout=60000)
        page.locator(".card").first.get_by_role(
            "button", name="View", exact=True
        ).click()
        expect(page.locator("#viewer-body")).to_contain_text("Reconstruction passed")
        expect(page.locator("#viewer-body")).to_contain_text("Solver: iterative")
        page.click("#close-viewer")
        page.select_option("#workflow", "filter")
        expect(page.locator("#preset")).to_have_value("filter-quick")
        expect(page.locator("#setting-degree")).to_have_attribute("min", "2")
        page.select_option("#preset", "interval-mixed")
        page.select_option("#workflow-filter", "interval_projector")
        page.click("#run")
        expect(page.locator(".card .status.completed")).to_have_count(1, timeout=60000)
        page.locator(".card").get_by_role("button", name="View", exact=True).click()
        expect(page.locator("#viewer-body")).to_contain_text("Solver failed")
        page.click("#close-viewer")
        page.select_option("#workflow", "hamiltonian_simulation")
        expect(page.locator("#setting-device_name")).to_have_attribute("type", "hidden")
        expect(page.locator("#setting-block_encoding")).to_have_attribute(
            "type", "hidden"
        )
        # Exercise finite-shot controls and report artifacts through the real UI.
        page.select_option("#workflow", "spectral_filter")
        page.select_option("#workflow-filter", "spectral_filter")
        page.select_option("#setting-shots", "100")
        page.select_option("#setting-input_state", '"basis-01"')
        page.click("#run")
        expect(page.locator(".card .status.completed")).to_have_count(1, timeout=60000)
        expect(page.locator(".card")).to_contain_text("100 shots")
        page.locator(".card").get_by_role("button", name="View", exact=True).click()
        expect(page.locator('#viewer-body img[src$="phases.png"]')).to_be_visible()
        expect(page.locator('#viewer-body img[src$="spectrum.png"]')).to_be_visible()
        check_image(page, "#viewer-body img")
        page.click("#close-viewer")
        # Cancel while the isolated worker starts, then check terminal filtering.
        page.click("#run")
        page.locator(".card").first.get_by_role(
            "button", name="Cancel", exact=True
        ).click()
        expect(page.locator(".card .status.cancelled")).to_have_count(1, timeout=30000)
        page.select_option("#status-filter", "active")
        expect(page.locator(".card")).to_have_count(0)
        page.select_option("#status-filter", "cancelled")
        expect(page.locator(".card")).to_have_count(1)
        page.reload()
        page.select_option("#status-filter", "cancelled")
        expect(page.locator(".card")).to_have_count(1)
        page.select_option("#status-filter", "")
        page.select_option("#workflow-filter", "")
        expect(page.locator(".card")).to_have_count(9)
        page.screenshot(path=str(args.screenshot), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(
            path=str(args.screenshot.with_stem(args.screenshot.stem + "-mobile")),
            full_page=True,
        )
        assert not errors, errors
        check_history_browser(browser, args.url)
        browser.close()
    print(
        "Browser smoke passed: run, viewer, export, exact reuse, comparison, "
        "incompatibility, favorites, sampling, report plots, cancellation, "
        "search, refresh, and mobile layout."
    )


if __name__ == "__main__":
    main()
