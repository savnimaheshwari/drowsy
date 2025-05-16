const video = document.getElementById("videoFeed");
video.src = "http://localhost:5050/video";

setInterval(() => {
  document.getElementById("statusText").textContent = "Status: Watching...";
}, 3000);
