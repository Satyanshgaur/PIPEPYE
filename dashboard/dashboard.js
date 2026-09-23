// PipePye Demonstration Dashboard JavaScript
document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("drop-zone");
    const dropZonePrompt = document.getElementById("drop-zone-prompt");
    const fileInput = document.getElementById("mps-file-input");
    const selectedFileInfo = document.getElementById("selected-file-info");
    const selectedFileName = document.getElementById("selected-file-name");
    const selectedFileSize = document.getElementById("selected-file-size");
    const btnClearFile = document.getElementById("btn-clear-file");
    const sampleSelect = document.getElementById("sample-select");
    const maxItersInput = document.getElementById("max-iters");
    const btnSolve = document.getElementById("btn-solve");
    const btnSpinner = document.getElementById("btn-spinner");
    const executionProgress = document.getElementById("execution-progress");
    const progressBarFill = document.getElementById("progress-bar-fill");
    const errorContainer = document.getElementById("error-container");
    const resultsView = document.getElementById("results-view");

    let currentFile = null;
    let currentSample = null;

    // Check System Status
    fetch("/api/status")
        .then(r => r.json())
        .then(data => {
            const dot = document.getElementById("system-status-dot");
            const text = document.getElementById("system-status-text");
            if (data.status === "ready") {
                dot.className = "status-dot ready";
                text.textContent = "Engine Ready (C++20/CUDA)";
            } else {
                dot.className = "status-dot";
                text.textContent = "Runner binary missing";
            }
        })
        .catch(() => {
            const text = document.getElementById("system-status-text");
            text.textContent = "Server offline";
        });

    // Populate Built-in Samples
    fetch("/api/samples")
        .then(r => r.json())
        .then(data => {
            if (!data.samples || data.samples.length === 0) return;
            
            // Group by category
            const groups = {};
            data.samples.forEach(s => {
                if (!groups[s.category]) groups[s.category] = [];
                groups[s.category].push(s);
            });

            sampleSelect.innerHTML = '<option value="">-- Choose from Netlib or Industrial Suite --</option>';
            for (const [category, items] of Object.entries(groups)) {
                const optgroup = document.createElement("optgroup");
                optgroup.label = category;
                items.forEach(s => {
                    const opt = document.createElement("option");
                    opt.value = JSON.stringify(s);
                    opt.textContent = `${s.name} (${formatBytes(s.size_bytes)})`;
                    optgroup.appendChild(opt);
                });
                sampleSelect.appendChild(optgroup);
            }
        })
        .catch(err => console.error("Failed to load samples:", err));

    // Dropzone / File Picker events
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    btnClearFile.addEventListener("click", (e) => {
        e.stopPropagation();
        resetFileSelection();
    });

    sampleSelect.addEventListener("change", () => {
        const val = sampleSelect.value;
        if (!val) {
            currentSample = null;
            if (!currentFile) btnSolve.disabled = true;
            return;
        }
        currentSample = JSON.parse(val);
        currentFile = null;
        fileInput.value = "";
        dropZonePrompt.style.display = "block";
        selectedFileInfo.style.display = "none";
        btnSolve.disabled = false;
    });

    function handleFileSelect(file) {
        if (!file.name.toLowerCase().endsWith(".mps")) {
            showError("Please select a valid .mps (Mathematical Programming System) file.");
            return;
        }
        hideError();
        currentFile = file;
        currentSample = null;
        sampleSelect.value = "";

        dropZonePrompt.style.display = "none";
        selectedFileInfo.style.display = "flex";
        selectedFileName.textContent = file.name;
        selectedFileSize.textContent = formatBytes(file.size);
        btnSolve.disabled = false;
    }

    function resetFileSelection() {
        currentFile = null;
        fileInput.value = "";
        dropZonePrompt.style.display = "block";
        selectedFileInfo.style.display = "none";
        if (!currentSample) btnSolve.disabled = true;
    }

    // Solve Trigger
    btnSolve.addEventListener("click", () => {
        if (!currentFile && !currentSample) return;

        hideError();
        resultsView.style.display = "none";
        btnSolve.disabled = true;
        btnSpinner.style.display = "inline-block";
        executionProgress.style.display = "block";

        // Progress animation
        let progress = 10;
        progressBarFill.style.width = `${progress}%`;
        setStepActive("step-ingest");

        const progressInterval = setInterval(() => {
            if (progress < 85) {
                progress += 15;
                progressBarFill.style.width = `${progress}%`;
                if (progress > 30) setStepActive("step-prep");
                if (progress > 50) setStepActive("step-struct");
                if (progress > 70) setStepActive("step-solve");
            }
        }, 120);

        const maxIters = parseInt(maxItersInput.value) || 3000;

        let requestPromise;
        if (currentFile) {
            // Upload user's chosen file
            const formData = new FormData();
            formData.append("file", currentFile);
            formData.append("max_iters", maxIters);
            requestPromise = fetch("/api/solve", {
                method: "POST",
                body: formData
            });
        } else if (currentSample) {
            // Use existing file path on Linux filesystem
            requestPromise = fetch("/api/solve", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    filepath: currentSample.abs_path,
                    max_iters: maxIters
                })
            });
        }

        requestPromise
            .then(async (res) => {
                clearInterval(progressInterval);
                const data = await res.json();
                if (!res.ok || data.error) {
                    throw new Error(data.error || "Solver execution encountered an error.");
                }
                return data;
            })
            .then((data) => {
                progressBarFill.style.width = "100%";
                setStepActive("step-verify");
                setTimeout(() => {
                    executionProgress.style.display = "none";
                    renderResults(data);
                }, 250);
            })
            .catch((err) => {
                clearInterval(progressInterval);
                executionProgress.style.display = "none";
                showError(`Execution Error: ${err.message}`);
            })
            .finally(() => {
                btnSolve.disabled = false;
                btnSpinner.style.display = "none";
            });
    });

    function setStepActive(stepId) {
        document.querySelectorAll(".step-label").forEach(el => el.classList.remove("active"));
        const target = document.getElementById(stepId);
        if (target) target.classList.add("active");
    }

    function showError(msg) {
        errorContainer.textContent = msg;
        errorContainer.style.display = "block";
    }

    function hideError() {
        errorContainer.style.display = "none";
        errorContainer.textContent = "";
    }

    function formatBytes(bytes) {
        if (!bytes || bytes === 0) return "0 B";
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(2) + " MB";
    }

    function formatSci(val, decimals = 4) {
        if (val === null || val === undefined) return "-";
        const num = Number(val);
        if (isNaN(num)) return "-";
        if (Math.abs(num) >= 1e4 || (Math.abs(num) > 0 && Math.abs(num) < 1e-3)) {
            return num.toExponential(decimals);
        }
        return num.toFixed(decimals);
    }

    // =========================================================================
    // RENDER RESULTS
    // =========================================================================
    function renderResults(data) {
        resultsView.style.display = "block";

        // Executive Summary
        document.getElementById("res-model-name").textContent = data.model?.name || "Unknown";
        document.getElementById("res-parse-time").textContent = `Parse: ${data.model?.parse_time_ms?.toFixed(2) || 0} ms`;
        
        const predEngine = `${data.phase5_structure_and_prediction?.predicted_solver || "-"} (${data.phase5_structure_and_prediction?.predicted_backend || "-"})`;
        document.getElementById("res-predicted-engine").textContent = predEngine;
        document.getElementById("res-prediction-rationale").textContent = data.phase5_structure_and_prediction?.prediction_rationale || "-";

        const winnerEngine = `${data.executive_summary?.actual_winner_solver || "-"} (${data.executive_summary?.actual_winner_backend || "-"})`;
        document.getElementById("res-winner-engine").textContent = winnerEngine;
        document.getElementById("res-winner-time").textContent = `Time: ${data.executive_summary?.best_runtime_ms?.toFixed(2) || 0} ms`;

        const badgeOutcome = document.getElementById("res-prediction-badge");
        const outcomeStr = data.executive_summary?.prediction_outcome || "UNKNOWN";
        badgeOutcome.textContent = outcomeStr;
        badgeOutcome.className = `badge badge-outcome ${outcomeStr.toLowerCase()}`;

        const isVerifValid = data.solution_verification?.is_valid;
        const verifBadge = document.getElementById("res-verif-badge");
        verifBadge.textContent = isVerifValid ? "Verification: PASSED" : "Verification: FAILED";
        verifBadge.style.color = isVerifValid ? "var(--color-success)" : "var(--color-danger)";

        document.getElementById("res-best-obj").textContent = formatSci(data.executive_summary?.best_objective, 6);

        // HiGHS Reference Banner Status
        const refVerif = data.phase6_reference_verification || {};
        const highsStatusEl = document.getElementById("res-highs-status");
        const highsGapEl = document.getElementById("res-highs-gap");
        if (refVerif.has_reference) {
            if (refVerif.is_verified_optimal) {
                highsStatusEl.textContent = "VERIFIED MATCH";
                highsStatusEl.style.color = "var(--color-success)";
                const gapPct = (refVerif.relative_gap_vs_highs * 100).toFixed(4);
                highsGapEl.textContent = `Gap vs HiGHS: ${gapPct}%`;
            } else if (refVerif.model_status === "Infeasible") {
                highsStatusEl.textContent = "HIGHS INFEASIBLE";
                highsStatusEl.style.color = "var(--color-warning)";
                highsGapEl.textContent = "Consistent Status";
            } else {
                highsStatusEl.textContent = "GAP DETECTED";
                highsStatusEl.style.color = "var(--color-warning)";
                const gapPct = refVerif.relative_gap_vs_highs !== null ? (refVerif.relative_gap_vs_highs * 100).toFixed(4) + "%" : "N/A";
                highsGapEl.textContent = `Gap: ${gapPct}`;
            }
        } else {
            highsStatusEl.textContent = "No Reference";
            highsStatusEl.style.color = "var(--text-muted)";
            highsGapEl.textContent = "Netlib / Custom Model";
        }

        // Phase 1: Sparse Core
        const p1 = data.phase1_sparse_core || {};
        document.getElementById("p1-dims").textContent = `${p1.raw_rows || 0} × ${p1.raw_cols || 0}`;
        document.getElementById("p1-nnz").textContent = (p1.raw_nonzeros || 0).toLocaleString();
        document.getElementById("p1-density").textContent = `${((p1.raw_density || 0) * 100).toFixed(4)} %`;
        document.getElementById("p1-vars").textContent = `${p1.continuous_vars || 0} Continuous, ${p1.binary_vars || 0} Binary, ${p1.integer_vars || 0} General Integer`;
        document.getElementById("p1-senses").textContent = `≤: ${p1.less_equal_rows || 0}, ≥: ${p1.greater_equal_rows || 0}, =: ${p1.equal_rows || 0}, Ranged: ${p1.range_rows || 0}`;
        document.getElementById("p1-coo-mem").textContent = formatBytes(p1.coo_bytes);
        document.getElementById("p1-csr-mem").textContent = formatBytes(p1.csr_bytes);
        document.getElementById("p1-csc-mem").textContent = formatBytes(p1.csc_bytes);
        document.getElementById("p1-spmv-suit").textContent = p1.spmv_suitability || "Standard CSR";

        // Phase 2: Presolve & Scaling
        const p2 = data.phase2_model_preparation || {};
        document.getElementById("p2-raw-rows").textContent = (p1.raw_rows || 0).toLocaleString();
        document.getElementById("p2-prep-rows").textContent = (p2.prepared_rows || 0).toLocaleString();
        document.getElementById("p2-rows-red").textContent = `-${p2.rows_removed || 0}`;

        document.getElementById("p2-raw-cols").textContent = (p1.raw_cols || 0).toLocaleString();
        document.getElementById("p2-prep-cols").textContent = (p2.prepared_cols || 0).toLocaleString();
        document.getElementById("p2-cols-red").textContent = `-${p2.cols_removed || 0}`;

        document.getElementById("p2-raw-nnz").textContent = (p1.raw_nonzeros || 0).toLocaleString();
        document.getElementById("p2-prep-nnz").textContent = (p2.prepared_nonzeros || 0).toLocaleString();
        document.getElementById("p2-nnz-red").textContent = `-${p2.nnz_removed || 0} (${(p2.nnz_reduction_pct || 0).toFixed(1)}%)`;

        document.getElementById("p2-raw-density").textContent = `${((p1.raw_density || 0) * 100).toFixed(3)} %`;
        document.getElementById("p2-prep-density").textContent = `${((p2.prepared_density || 0) * 100).toFixed(3)} %`;
        const densityDelta = ((p2.prepared_density || 0) - (p1.raw_density || 0)) * 100;
        document.getElementById("p2-density-change").textContent = `${densityDelta >= 0 ? "+" : ""}${densityDelta.toFixed(3)} %`;

        document.getElementById("p2-dynamic-range").textContent = 
            `${formatSci(p2.dynamic_range_before, 1)} → ${formatSci(p2.dynamic_range_after, 1)}`;
        document.getElementById("p2-prep-time").textContent = `${(p2.prep_time_ms || 0).toFixed(2)} ms`;

        // Phase 5: Structure & Prediction
        const p5 = data.phase5_structure_and_prediction || {};
        document.getElementById("p5-staircase").textContent = (p5.staircase_score || 0).toFixed(3);
        document.getElementById("p5-row-gini").textContent = (p5.row_gini_index || 0).toFixed(3);
        document.getElementById("p5-row-degree").textContent = `${p5.max_row_degree || 0} / ${(p5.avg_row_degree || 0).toFixed(1)}`;
        document.getElementById("p5-bandwidth").textContent = (p5.half_bandwidth || 0).toLocaleString();
        document.getElementById("p5-integrality").textContent = `${((p5.integrality_ratio || 0) * 100).toFixed(1)} %`;
        document.getElementById("p5-rec-badge").textContent = `${p5.predicted_solver || "-"} on ${p5.predicted_backend || "-"}`;
        document.getElementById("p5-rec-rationale").textContent = p5.prediction_rationale || "-";

        // Phase 3: PDHG
        const pdhg = data.phase3_pdhg || {};
        const cpu = pdhg.cpu || {};
        const gpu = pdhg.gpu || {};

        document.getElementById("pdhg-cpu-status").textContent = cpu.status || "SKIPPED";
        document.getElementById("pdhg-cpu-status").className = `badge ${cpu.status === "OPTIMAL" ? "badge-success" : ""}`;
        document.getElementById("pdhg-cpu-obj").textContent = formatSci(cpu.objective, 4);
        document.getElementById("pdhg-cpu-iters").textContent = (cpu.iterations || 0).toLocaleString();
        document.getElementById("pdhg-cpu-pres").textContent = formatSci(cpu.primal_residual, 2);
        document.getElementById("pdhg-cpu-dres").textContent = formatSci(cpu.dual_residual, 2);
        document.getElementById("pdhg-cpu-time").textContent = `${(cpu.solve_time_ms || 0).toFixed(2)} ms`;

        document.getElementById("pdhg-gpu-status").textContent = gpu.status || "NOT_AVAILABLE";
        document.getElementById("pdhg-gpu-status").className = `badge ${gpu.status === "OPTIMAL" ? "badge-success" : ""}`;
        document.getElementById("pdhg-gpu-obj").textContent = formatSci(gpu.objective, 4);
        document.getElementById("pdhg-gpu-iters").textContent = (gpu.iterations || 0).toLocaleString();
        document.getElementById("pdhg-gpu-pres").textContent = formatSci(gpu.primal_residual, 2);
        document.getElementById("pdhg-gpu-dres").textContent = formatSci(gpu.dual_residual, 2);
        document.getElementById("pdhg-gpu-time").textContent = gpu.total_time_ms !== null ? `${gpu.total_time_ms.toFixed(2)} ms` : "null";

        const gpuBox = document.getElementById("gpu-timing-box");
        if (gpu.executed) {
            gpuBox.style.display = "flex";
            document.getElementById("chip-h2d").textContent = `H2D: ${(gpu.h2d_transfer_ms || 0).toFixed(2)} ms`;
            document.getElementById("chip-pure").textContent = `Kernel: ${(gpu.pure_solve_ms || 0).toFixed(2)} ms`;
            document.getElementById("chip-d2h").textContent = `D2H: ${(gpu.d2h_transfer_ms || 0).toFixed(2)} ms`;
        } else {
            gpuBox.style.display = "none";
        }

        // Render Convergence SVG Chart
        const historyData = (gpu.history && gpu.history.length > 0) ? gpu.history : (cpu.history || []);
        renderConvergenceSvg(historyData);

        // Phase 4: Dual Simplex & Crossover
        const ds = data.phase4_dual_simplex || {};
        document.getElementById("ds-status").textContent = ds.status || "SKIPPED";
        document.getElementById("ds-status").className = `badge ${ds.status === "OPTIMAL" ? "badge-success" : ""}`;
        document.getElementById("ds-obj").textContent = formatSci(ds.objective, 6);
        document.getElementById("ds-pivots").textContent = (ds.pivots || 0).toLocaleString();
        document.getElementById("ds-bound-flips").textContent = (ds.bound_flips || 0).toLocaleString();
        document.getElementById("ds-factorizations").textContent = (ds.factorizations || 0).toLocaleString();
        document.getElementById("ds-eta-updates").textContent = (ds.pfi_eta_updates || 0).toLocaleString();
        document.getElementById("ds-tran-counts").textContent = `${ds.ftran_count || 0} / ${ds.btran_count || 0}`;
        document.getElementById("ds-time").textContent = `${(ds.solve_time_ms || 0).toFixed(2)} ms`;

        const co = data.phase4_crossover || {};
        document.getElementById("co-status").textContent = co.succeeded ? "OPTIMAL VERTEX" : (co.executed ? "FAILED" : "SKIPPED");
        document.getElementById("co-status").className = `badge ${co.succeeded ? "badge-success" : ""}`;
        document.getElementById("co-bounds").textContent = (co.active_bounds_detected || 0).toLocaleString();
        document.getElementById("co-partition").textContent = `${co.structural_basic_vars || 0} Struct / ${co.slack_basic_vars || 0} Slack`;
        document.getElementById("co-pivots").textContent = (co.cleanup_pivots || 0).toLocaleString();
        document.getElementById("co-time").textContent = `${(co.total_crossover_time_ms || 0).toFixed(2)} ms`;
        document.getElementById("co-obj").textContent = formatSci(co.final_objective, 6);

        // Phase 5: MILP Branch-and-Bound
        const milp = data.phase5_milp || {};
        const milpCardApplicable = document.getElementById("milp-applicable");
        const milpCardPlaceholder = document.getElementById("milp-not-applicable");

        if (milp.is_milp && milp.executed) {
            milpCardApplicable.style.display = "block";
            milpCardPlaceholder.style.display = "none";
            document.getElementById("milp-warm-pivots").textContent = (milp.warm_pivots || 0).toLocaleString();
            document.getElementById("milp-warm-nodes").textContent = (milp.warm_nodes_explored || 0).toLocaleString();
            document.getElementById("milp-warm-time").textContent = `${(milp.warm_time_ms || 0).toFixed(2)} ms`;

            document.getElementById("milp-cold-pivots").textContent = (milp.cold_pivots || 0).toLocaleString();
            document.getElementById("milp-cold-nodes").textContent = (milp.cold_nodes_explored || 0).toLocaleString();
            document.getElementById("milp-cold-time").textContent = `${(milp.cold_time_ms || 0).toFixed(2)} ms`;

            document.getElementById("milp-reduction-pct").textContent = `${(milp.pivot_reduction_pct || 0).toFixed(1)} % Pivot Reduction`;
            document.getElementById("milp-best-obj").textContent = formatSci(milp.integer_objective, 6);
        } else {
            milpCardApplicable.style.display = "none";
            milpCardPlaceholder.style.display = "block";
        }

        // Phase 6: HiGHS Reference Verification & Provenance
        const p6Applicable = document.getElementById("phase6-applicable");
        const p6Placeholder = document.getElementById("phase6-placeholder");
        const drawer = document.getElementById("formulation-drawer");
        if (drawer) drawer.style.display = "none"; // Reset drawer

        if (refVerif.has_reference || (data.workload_provenance && data.workload_provenance.has_provenance)) {
            p6Applicable.style.display = "grid";
            p6Placeholder.style.display = "none";

            const certBadge = document.getElementById("highs-cert-badge");
            if (refVerif.is_verified_optimal) {
                certBadge.textContent = "Certified Match (< 1e-4 gap)";
                certBadge.className = "badge badge-success";
            } else if (refVerif.model_status === "Infeasible") {
                certBadge.textContent = "Status: Infeasible (Matches HiGHS)";
                certBadge.className = "badge badge-warning";
            } else {
                certBadge.textContent = "Reference Status: " + (refVerif.model_status || "Known");
                certBadge.className = "badge badge-info";
            }

            // PipePye vs HiGHS comparison table
            const isMilp = data.phase5_milp?.is_milp;
            document.getElementById("comp-pipepye-status").textContent = 
                (isMilp && data.phase5_milp?.executed) ? "OPTIMAL" : 
                (data.phase4_dual_simplex?.status || "OPTIMAL");
            document.getElementById("comp-highs-status").textContent = refVerif.model_status || "Optimal";

            const pipeObj = data.executive_summary?.best_objective;
            const highsObj = refVerif.reference_objective;
            document.getElementById("comp-pipepye-obj").textContent = formatSci(pipeObj, 6);
            document.getElementById("comp-highs-obj").textContent = formatSci(highsObj, 6);

            const relGapEl = document.getElementById("comp-rel-gap");
            if (refVerif.relative_gap_vs_highs !== null && refVerif.relative_gap_vs_highs !== undefined) {
                const gapPct = (refVerif.relative_gap_vs_highs * 100).toFixed(4);
                relGapEl.textContent = `${gapPct}% relative gap (Verified ${refVerif.is_verified_optimal ? "OPTIMAL" : "CLOSE"})`;
                relGapEl.className = refVerif.is_verified_optimal ? "mono text-green font-bold" : "mono text-pink font-bold";
            } else {
                relGapEl.textContent = "N/A (Infeasible or unbounded)";
                relGapEl.className = "mono text-muted";
            }

            // Pivots / Iterations
            const pipePivots = isMilp ? (data.phase5_milp?.warm_pivots || 0) : (data.phase4_dual_simplex?.pivots || 0);
            document.getElementById("comp-pipepye-pivots").textContent = pipePivots.toLocaleString();
            document.getElementById("comp-highs-iters").textContent = (refVerif.highs_simplex_iterations || 0).toLocaleString();

            // Nodes
            const pipeNodes = isMilp ? (data.phase5_milp?.warm_nodes_explored || 0) : 0;
            const highsNodes = refVerif.highs_mip_nodes >= 0 ? refVerif.highs_mip_nodes : 0;
            document.getElementById("comp-pipepye-nodes").textContent = isMilp ? pipeNodes.toLocaleString() : "N/A (Continuous LP)";
            document.getElementById("comp-highs-nodes").textContent = isMilp ? highsNodes.toLocaleString() : "N/A (Continuous LP)";

            // Runtime
            const pipeTimeMs = data.executive_summary?.best_runtime_ms || 0;
            const highsTimeMs = (refVerif.highs_wallclock_sec || 0) * 1000;
            document.getElementById("comp-pipepye-time").textContent = `${pipeTimeMs.toFixed(2)} ms`;
            document.getElementById("comp-highs-time").textContent = `${highsTimeMs.toFixed(2)} ms`;

            // Industrial Provenance
            const prov = data.workload_provenance || {};
            document.getElementById("prov-form-class").textContent = prov.formulation_class || (isMilp ? "MILP" : "LP");
            document.getElementById("prov-category").textContent = prov.category || "Industrial Energy & Refining";
            document.getElementById("prov-benchmark").textContent = prov.provenance_benchmark || "Academic Literature Reference";
            document.getElementById("prov-structure").textContent = prov.mathematical_structure || "Sparse Structured Industrial System";

            const rng = prov.ranges || {};
            if (rng.min_abs_coeff !== undefined) {
                document.getElementById("prov-ranges").textContent = 
                    `|A| ∈ [${rng.min_abs_coeff}, ${rng.max_abs_coeff}], RHS ∈ [${rng.min_rhs}, ${rng.max_rhs}], Dynamic Range: ${prov.structural_metrics?.dynamic_range || "-"}`;
            } else {
                document.getElementById("prov-ranges").textContent = `Dynamic Range: ${(p2.dynamic_range_before || 0).toFixed(1)}`;
            }

            // Setup Formulation Toggle Button
            const btnForm = document.getElementById("btn-toggle-formulation");
            if (btnForm) {
                btnForm.onclick = () => {
                    if (drawer.style.display === "block") {
                        drawer.style.display = "none";
                        return;
                    }
                    const caseId = prov.case_id || refVerif.case_id;
                    if (!caseId) return;

                    fetch(`/api/formulation?case=${encodeURIComponent(caseId)}`)
                        .then(r => r.json())
                        .then(fData => {
                            if (fData.markdown) {
                                document.getElementById("formulation-markdown-content").innerHTML = renderSimpleMarkdown(fData.markdown);
                                document.getElementById("formulation-doc-title").textContent = `Mathematical Formulation: ${prov.name || caseId}`;
                                drawer.style.display = "block";
                                drawer.scrollIntoView({ behavior: "smooth" });
                            }
                        })
                        .catch(err => {
                            console.error("Failed to load formulation:", err);
                        });
                };
            }

            const btnCloseForm = document.getElementById("btn-close-formulation");
            if (btnCloseForm) {
                btnCloseForm.onclick = () => {
                    drawer.style.display = "none";
                };
            }

        } else {
            p6Applicable.style.display = "none";
            p6Placeholder.style.display = "block";
        }

        // Verification Audit
        const verif = data.solution_verification || {};
        document.getElementById("verif-engine").textContent = verif.audited_solver || "-";
        const feasBadge = document.getElementById("verif-feas-badge");
        feasBadge.textContent = verif.is_feasible ? "FEASIBLE" : "INFEASIBLE";
        feasBadge.className = `badge ${verif.is_feasible ? "badge-success" : "badge-danger"}`;
        document.getElementById("verif-bound-viol").textContent = formatSci(verif.max_bound_violation, 2);
        document.getElementById("verif-row-viol").textContent = formatSci(verif.max_constraint_violation, 2);
        document.getElementById("verif-obj-mismatch").textContent = formatSci(verif.objective_mismatch, 2);

        // Smooth scroll to summary
        resultsView.scrollIntoView({ behavior: "smooth" });
    }

    // =========================================================================
    // CONVERGENCE SVG CHART RENDERING
    // =========================================================================
    function renderConvergenceSvg(history) {
        const svg = document.getElementById("convergence-svg");
        svg.innerHTML = "";

        if (!history || history.length === 0) {
            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("x", "250");
            text.setAttribute("y", "110");
            text.setAttribute("text-anchor", "middle");
            text.setAttribute("fill", "#64748b");
            text.textContent = "No iteration convergence history recorded.";
            svg.appendChild(text);
            return;
        }

        const width = 500;
        const height = 220;
        const padLeft = 50;
        const padRight = 20;
        const padTop = 20;
        const padBottom = 30;

        const chartW = width - padLeft - padRight;
        const chartH = height - padTop - padBottom;

        // Find iteration range
        const iters = history.map(h => h.iter);
        const minIter = Math.min(...iters);
        const maxIter = Math.max(...iters);

        // Log10 range for residuals
        const logValues = [];
        history.forEach(h => {
            if (h.primal > 0) logValues.push(Math.log10(h.primal));
            if (h.dual > 0) logValues.push(Math.log10(h.dual));
        });

        const minLog = logValues.length > 0 ? Math.floor(Math.min(...logValues)) : -12;
        const maxLog = logValues.length > 0 ? Math.ceil(Math.max(...logValues)) : 2;

        const xCoord = (iter) => {
            if (maxIter === minIter) return padLeft;
            return padLeft + ((iter - minIter) / (maxIter - minIter)) * chartW;
        };

        const yCoord = (val) => {
            if (val <= 0) val = 1e-16;
            const logVal = Math.log10(val);
            const clamped = Math.max(minLog, Math.min(maxLog, logVal));
            return padTop + ((maxLog - clamped) / (maxLog - minLog)) * chartH;
        };

        // Render Grid Lines & Y-ticks
        const numTicks = Math.min(6, maxLog - minLog);
        const step = Math.max(1, Math.floor((maxLog - minLog) / numTicks));

        for (let l = minLog; l <= maxLog; l += step) {
            const y = padTop + ((maxLog - l) / (maxLog - minLog)) * chartH;
            
            // Grid line
            const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
            line.setAttribute("x1", padLeft);
            line.setAttribute("x2", width - padRight);
            line.setAttribute("y1", y);
            line.setAttribute("y2", y);
            line.setAttribute("stroke", "#334155");
            line.setAttribute("stroke-dasharray", "3 3");
            line.setAttribute("stroke-width", "0.8");
            svg.appendChild(line);

            // Tick label
            const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
            label.setAttribute("x", padLeft - 8);
            label.setAttribute("y", y + 3);
            label.setAttribute("text-anchor", "end");
            label.setAttribute("fill", "#64748b");
            label.setAttribute("font-size", "10");
            label.setAttribute("font-family", "monospace");
            label.textContent = `1e${l}`;
            svg.appendChild(label);
        }

        // X-Axis Line
        const xAxis = document.createElementNS("http://www.w3.org/2000/svg", "line");
        xAxis.setAttribute("x1", padLeft);
        xAxis.setAttribute("x2", width - padRight);
        xAxis.setAttribute("y1", height - padBottom);
        xAxis.setAttribute("y2", height - padBottom);
        xAxis.setAttribute("stroke", "#475569");
        xAxis.setAttribute("stroke-width", "1");
        svg.appendChild(xAxis);

        // X-Axis Labels
        const xMinText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        xMinText.setAttribute("x", padLeft);
        xMinText.setAttribute("y", height - 10);
        xMinText.setAttribute("text-anchor", "start");
        xMinText.setAttribute("fill", "#64748b");
        xMinText.setAttribute("font-size", "10");
        xMinText.textContent = `iter ${minIter}`;
        svg.appendChild(xMinText);

        const xMaxText = document.createElementNS("http://www.w3.org/2000/svg", "text");
        xMaxText.setAttribute("x", width - padRight);
        xMaxText.setAttribute("y", height - 10);
        xMaxText.setAttribute("text-anchor", "end");
        xMaxText.setAttribute("fill", "#64748b");
        xMaxText.setAttribute("font-size", "10");
        xMaxText.textContent = `iter ${maxIter}`;
        svg.appendChild(xMaxText);

        // Polylines for Primal and Dual residuals
        let primalPoints = [];
        let dualPoints = [];

        history.forEach(h => {
            const x = xCoord(h.iter);
            const yP = yCoord(h.primal);
            const yD = yCoord(h.dual);
            primalPoints.push(`${x.toFixed(1)},${yP.toFixed(1)}`);
            dualPoints.push(`${x.toFixed(1)},${yD.toFixed(1)}`);
        });

        // Dual residual polyline (pink)
        const polyDual = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        polyDual.setAttribute("points", dualPoints.join(" "));
        polyDual.setAttribute("fill", "none");
        polyDual.setAttribute("stroke", "#ec4899");
        polyDual.setAttribute("stroke-width", "1.8");
        polyDual.setAttribute("stroke-linecap", "round");
        svg.appendChild(polyDual);

        // Primal residual polyline (blue)
        const polyPrimal = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
        polyPrimal.setAttribute("points", primalPoints.join(" "));
        polyPrimal.setAttribute("fill", "none");
        polyPrimal.setAttribute("stroke", "#38bdf8");
        polyPrimal.setAttribute("stroke-width", "1.8");
        polyPrimal.setAttribute("stroke-linecap", "round");
        svg.appendChild(polyPrimal);
    }

    // =========================================================================
    // LIGHTWEIGHT MARKDOWN TO HTML RENDERER (FOR FORMULATION VIEWER)
    // =========================================================================
    function renderSimpleMarkdown(md) {
        if (!md) return "";
        let lines = md.split("\n");
        let html = [];
        let inCode = false;
        let inTable = false;
        let tableRows = [];

        for (let i = 0; i < lines.length; ++i) {
            let line = lines[i];

            if (line.trim().startsWith("```")) {
                if (inCode) {
                    html.push("</code></pre>");
                    inCode = false;
                } else {
                    html.push("<pre><code>");
                    inCode = true;
                }
                continue;
            }

            if (inCode) {
                html.push(escapeHtml(line) + "\n");
                continue;
            }

            // Table parsing
            if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
                if (!inTable) {
                    inTable = true;
                    tableRows = [];
                }
                if (!/^\|[\s\-:]+(\|[\s\-:]+)+\|$/.test(line.trim())) {
                    tableRows.push(line.trim());
                }
                continue;
            } else if (inTable) {
                inTable = false;
                html.push(renderTable(tableRows));
                tableRows = [];
            }

            // Headings
            if (line.startsWith("### ")) {
                html.push(`<h3>${formatInline(line.slice(4))}</h3>`);
            } else if (line.startsWith("## ")) {
                html.push(`<h2>${formatInline(line.slice(3))}</h2>`);
            } else if (line.startsWith("# ")) {
                html.push(`<h1>${formatInline(line.slice(2))}</h1>`);
            } else if (line.trim().startsWith("- ")) {
                html.push(`<ul><li>${formatInline(line.trim().slice(2))}</li></ul>`);
            } else if (/^\d+\.\s/.test(line.trim())) {
                let text = line.trim().replace(/^\d+\.\s/, "");
                html.push(`<ol><li>${formatInline(text)}</li></ol>`);
            } else if (line.trim().startsWith("$$") && line.trim().endsWith("$$") && line.trim().length > 4) {
                html.push(`<pre class="math"><code>${escapeHtml(line.trim().slice(2, -2).trim())}</code></pre>`);
            } else if (line.trim() === "") {
                html.push("");
            } else {
                html.push(`<p>${formatInline(line)}</p>`);
            }
        }

        if (inTable) {
            html.push(renderTable(tableRows));
        }

        return html.join("\n")
            .replace(/<\/ul>\s*<ul>/g, "")
            .replace(/<\/ol>\s*<ol>/g, "");
    }

    function escapeHtml(text) {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    function formatInline(text) {
        let s = escapeHtml(text);
        s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        s = s.replace(/\*(.*?)\*/g, '<em>$1</em>');
        s = s.replace(/`([^`]+)`/g, '<code>$1</code>');
        s = s.replace(/\$\$([\s\S]*?)\$\$/g, '<code>$1</code>');
        s = s.replace(/\$([^\$]+)\$/g, '<code>$1</code>');
        return s;
    }

    function renderTable(rows) {
        if (!rows || rows.length === 0) return "";
        let out = ["<table class=\"data-table\">"];
        for (let r = 0; r < rows.length; ++r) {
            let cells = rows[r].split("|").slice(1, -1);
            let tag = (r === 0) ? "th" : "td";
            let rowHtml = "<tr>" + cells.map(c => `<${tag}>${formatInline(c.trim())}</${tag}>`).join("") + "</tr>";
            if (r === 0) {
                out.push(`<thead>${rowHtml}</thead><tbody>`);
            } else {
                out.push(rowHtml);
            }
        }
        out.push("</tbody></table>");
        return out.join("");
    }
});
