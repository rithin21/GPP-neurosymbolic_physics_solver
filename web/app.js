const problem = document.querySelector("#problem");
const overlay = document.querySelector("#overlay");
const solveButton = document.querySelector("#solve");
const answer = document.querySelector("#answer");
const extraction = document.querySelector("#extraction");
const graph = document.querySelector("#graph");
const weak = document.querySelector("#weak");

async function refreshWeakNodes() {
  const response = await fetch("/api/weak-nodes");
  const data = await response.json();
  weak.textContent = data.length ? JSON.stringify(data, null, 2) : "No attempts yet.";
}

solveButton.addEventListener("click", async () => {
  answer.textContent = "Solving...";
  const response = await fetch("/api/solve", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      text: problem.value,
      overlay_id: overlay.value,
      correct: true
    })
  });

  if (!response.ok) {
    const error = await response.json();
    answer.textContent = error.detail || "Solve request failed.";
    return;
  }

  const data = await response.json();
  answer.textContent = data.narration;
  extraction.textContent = JSON.stringify(data.extraction, null, 2);
  graph.textContent = JSON.stringify({
    constraints: data.applied_constraints,
    steps: data.steps,
    law_nodes: data.law_nodes,
    retrieved_cases: data.retrieved_cases,
    unit_validation: data.unit_validation,
    phase_trace: data.phase_trace
  }, null, 2);
  await refreshWeakNodes();
});

refreshWeakNodes();
