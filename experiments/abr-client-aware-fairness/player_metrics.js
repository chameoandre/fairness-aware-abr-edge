// === Player configurations === 

function applyDashSettings(player, settings) {
    if (!player || typeof player.updateSettings !== "function") {
        console.error("Dash.js player not initialized correctly.");
        return;
    }

    player.updateSettings(settings);

    console.log("[CONFIG] Applied settings:", settings);
}

// function configureDashPlayer(player, abr_mode) {
//     if (!player || typeof player.updateSettings !== "function") {
//         console.error("Dash.js player not initialized correctly.");
//         return;
//     }

//     let abrSettings;
//     if (abr_mode === "DASH-DEFAULT" || abr_mode === "DASH-QOE-BASED") {
//         abrSettings = { enabled: true };
//     } 
//     else if (abr_mode === "BOLA") {
//         abrSettings = {
//             useDefaultABRRules: false,
//             ABRStrategy: 'abrBola'
//         };
//     } 
//     else {
//         // Todos os modos RL desligam o ABR para controle externo
//         abrSettings = { enabled: false };
//     }

//     player.updateSettings({
//         streaming: {
//             abr: abrSettings,
//             buffer: {
//                 initialBufferTime: 2,
//                 stableBufferTime: 10,
//                 bufferTimeAtTopQuality: 20,
//                 bufferTimeAtTopQualityLongForm: 30
//             }
//         }
//     });

//     console.log(`Dash.js player configured for ABR mode: ${abr_mode}`);
// }


function injectPlayerEventListeners() {
    // == LISTENER REINJECTION PROTECTION ==
    if (window.hasInjectedBufferListeners) {
        return;  // Já injetado, evita reinjeção
    }
    window.hasInjectedBufferListeners = true;
    
    // == BUFFERING EVENTS ==
    if (!window.bufferingEvents) {
        window.bufferingEvents = [];
        window.lastBufferStallStart = null;
    }

    // === DEFINE HANDLERS (always) ===
    const startStall = () => {
        if (window.lastBufferStallStart === null) {
            window.lastBufferStallStart = performance.now();
        }
    };
    
    const endStall = () => {
        if (window.lastBufferStallStart !== null) {
            let stallDuration = performance.now() - window.lastBufferStallStart;
            window.bufferingEvents.push(stallDuration);
            window.lastBufferStallStart = null;
        }
    };

        // === REMOVED AFTER MODIFICATION TO CAPTURE BUFFER === 

        // player.on(dashjs.MediaPlayer.events.PLAYBACK_STALLED, function () {
        //     window.lastBufferStallStart = performance.now();
        // });

        // player.on(dashjs.MediaPlayer.events.PLAYBACK_PLAYING, function () {
        //     if (window.lastBufferStallStart !== null) {
        //         let stallDuration = performance.now() - window.lastBufferStallStart;
        //         window.bufferingEvents.push(stallDuration);
        //         window.lastBufferStallStart = null;
        //     }
        // });

    // === REGISTER ALL RELEVANT BUFFER EVENTS  === 
    player.on(dashjs.MediaPlayer.events.PLAYBACK_STALLED, startStall);
    player.on(dashjs.MediaPlayer.events.PLAYBACK_PLAYING, endStall);

    player.on(dashjs.MediaPlayer.events.BUFFER_EMPTY, startStall);
    player.on(dashjs.MediaPlayer.events.BUFFER_LOADED, endStall);


    //  == STARTUP DELAY ==
    if (typeof window.startupDelay === "undefined" || typeof window.startupDelay !== "number") {
        window.playClickedAt = performance.now();

        player.on(dashjs.MediaPlayer.events.PLAYBACK_STARTED, function () {
            if ((typeof window.startupDelay === "undefined" || typeof window.startupDelay !== "number") 
                && typeof window.playClickedAt !== "undefined") {
                const delayMs = performance.now() - window.playClickedAt;
                window.startupDelay = Math.max(0, delayMs);
                // console.log("✅ Startup delay recorded (ms):", window.startupDelay);
            }
        });
    }

    // == SEGMENT DOWNLOAD EVENT (used to trigger metric sampling) ==

    // if (typeof window.metricReady === "undefined") {
    //     window.metricReady = false;

    //     player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, function (e) {
    //         if (e.request && e.request.mediaType === "video") {
    //             window.metricReady = true;
    //         }
    //     });
    // }

    // == SEGMENT COUNTER ==
    if (typeof window.segmentCounter === "undefined") {
        window.segmentCounter = 0;
    
        player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, function (e) {
            if (e.request.mediaType === "video") {
                window.segmentCounter += 1;
            }
        });
    }

}

function injectPlayerEventListeners_ORIGINAL_WORKING() {
    // == BUFFERING EVENTS ==
    if (!window.bufferingEvents) {
        window.bufferingEvents = [];
        window.lastBufferStallStart = null;

        player.on(dashjs.MediaPlayer.events.PLAYBACK_STALLED, function () {
            window.lastBufferStallStart = performance.now();
        });

        player.on(dashjs.MediaPlayer.events.PLAYBACK_PLAYING, function () {
            if (window.lastBufferStallStart !== null) {
                let stallDuration = performance.now() - window.lastBufferStallStart;
                window.bufferingEvents.push(stallDuration);
                window.lastBufferStallStart = null;
            }
        });
    }

    //  == STARTUP DELAY ==
    if (typeof window.startupDelay === "undefined" || typeof window.startupDelay !== "number") {
        window.playClickedAt = performance.now();

        player.on(dashjs.MediaPlayer.events.PLAYBACK_STARTED, function () {
            if ((typeof window.startupDelay === "undefined" || typeof window.startupDelay !== "number") 
                && typeof window.playClickedAt !== "undefined") {
                const delayMs = performance.now() - window.playClickedAt;
                window.startupDelay = Math.max(0, delayMs);
                // console.log("✅ Startup delay recorded (ms):", window.startupDelay);
            }
        });
    }

    // == SEGMENT DOWNLOAD EVENT (used to trigger metric sampling) ==

    // if (typeof window.metricReady === "undefined") {
    //     window.metricReady = false;

    //     player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, function (e) {
    //         if (e.request && e.request.mediaType === "video") {
    //             window.metricReady = true;
    //         }
    //     });
    // }

    // == SEGMENT COUNTER ==
    if (typeof window.segmentCounter === "undefined") {
        window.segmentCounter = 0;
    
        player.on(dashjs.MediaPlayer.events.FRAGMENT_LOADING_COMPLETED, function (e) {
            if (e.request.mediaType === "video") {
                window.segmentCounter += 1;
            }
        });
    }
}




// player_metrics.js

// function captureStartupDelay(player) {
//     return new Promise((resolve, reject) => {
//         if (!player) {
//             reject("Player não inicializado");
//             return;
//         }

//         const startTime = performance.now();

//         // Adiciona o listener para o evento PLAYBACK_STARTED
//         const onPlaybackStarted = function() {
//             const endTime = performance.now();
//             const startupDelay = (endTime - startTime) / 1000; // Em segundos
//             resolve(startupDelay);  // Resolve a Promise com o valor do startup delay

//             // Remove o listener após o evento ser disparado
//             player.off(dashjs.MediaPlayer.events.PLAYBACK_STARTED, onPlaybackStarted);
//         };

//         // Adiciona o listener
//         player.on(dashjs.MediaPlayer.events.PLAYBACK_STARTED, onPlaybackStarted);
//     });
// }

window.getTransferLatency = function () {
    try {
        const resources = performance.getEntriesByType('resource').filter(
            r => r.name.includes('.mpd') || r.name.includes('.m4s')
        );
        if (resources.length > 0) {
            const latest = resources[resources.length - 1];
            return latest.responseEnd - latest.startTime;
        }
        return 0;
    } catch (e) {
        return 0;
    }
}

// window.getLatency_OLD = function () {
//     try {
//         var requests = player.getDashMetrics().getHttpRequests('video');
//         var lastRequest = requests && requests.length > 0 ? requests[requests.length - 1] : null;

//         if (lastRequest && lastRequest.trequest && lastRequest.tresponse) {
//             return lastRequest.tresponse - lastRequest.trequest;
//         }
//         return 0;
//     } catch (e) {
//         return 0;
//     }
// }


window.getLatency = function () {
    try {
        var requests = player.getDashMetrics().getHttpRequests('video');
        if (!requests || requests.length === 0) return 0;

        // Filtra somente requests válidos (2xx ou 3xx)
        var validRequests = requests.filter(function (req) {
            return req.trequest && req.tresponse &&
                   req.responsecode >= 200 && req.responsecode < 400;
        });

        if (validRequests.length === 0) return 0;

        var lastRequest = validRequests[validRequests.length - 1];
        var latency = lastRequest.tresponse - lastRequest.trequest;

        if (latency < 0) {
            console.warn('Negative latency detected:', lastRequest);
            return 0;
        }

        return latency;
    } catch (e) {
        console.error('Error computing latency:', e);
        return 0;
    }
}



function getAvgThroughput() {
    //Returning metric in Bps (bits per second)
    const throughput = player.getAverageThroughput('video');
    // return player.getAverageThroughput('video');
    return (typeof throughput === 'number' && !isNaN(throughput)) ? throughput : 0;
}

window.getStartupDelay = function () {
    try {
        const playbackMetrics = player.getMetricsFor('video');
        if (playbackMetrics && playbackMetrics.PlayList && playbackMetrics.PlayList.length > 0) {
            const playlist = playbackMetrics.PlayList[0];
            if (playlist.start !== null && playlist.start !== undefined &&
                playlist.request && playlist.request.startTime !== undefined) {
                const startupDelay = playlist.start - playlist.request.startTime;
                return (typeof startupDelay === 'number' && !isNaN(startupDelay)) ? startupDelay : 0;
            }
        }
        return 0;
    } catch (e) {
        console.error("Error retrieving startup delay:", e);
        return 0;
    }
}


function getCurrentBufferLevel() {
    var dashMetrics = player.getDashMetrics();
    var bufferLevel = dashMetrics.getCurrentBufferLevel('video');
    return bufferLevel ? bufferLevel : 0;
}

function getDroppedFrames() {
    var dashMetrics = player.getDashMetrics();
    var droppedFrames = dashMetrics.getCurrentDroppedFrames('video');
    return droppedFrames ? droppedFrames.droppedFrames : 0;
}

function getQuality() {
    // return player.getQualityFor('video');
    const quality = player.getQualityFor('video');
    return (typeof quality === 'number' && !isNaN(quality)) ? quality : 0;
}

//Returning true if "buffering" and false if bufferstate was null
function isBuffering() {
    // var dashMetrics = player.getDashMetrics();
    // var bufferingState = dashMetrics.getCurrentBufferState('video');
    // return bufferingState.state ? bufferingState.state : null;
    const dashMetrics = player.getDashMetrics();
    const bufferState = dashMetrics.getCurrentBufferState('video');
    // return bufferState && bufferState.state === "buffering";
    return bufferState && bufferState.state ? bufferState.state : null;
}

function getBufferEvents() {
    var dashMetrics = player.getDashMetrics();
    return dashMetrics.getCurrentBufferState('video').state === "buffering";
}

function getBufferLength() {
    // var dashMetrics = player.getDashMetrics();
    // return dashMetrics.getCurrentBufferLevel('video');
    const dashMetrics = player.getDashMetrics();
    const bufferLevel = dashMetrics.getCurrentBufferLevel('video');
    return (typeof bufferLevel === 'number' && !isNaN(bufferLevel)) ? bufferLevel : 0;
}


function getCurrentPlaybackTime(player){
    // return player.getVideoElement().currentTime;
    const videoElement = player.getVideoElement();
    return (videoElement && typeof videoElement.currentTime === 'number') ? videoElement.currentTime : 0;
}


window.hasDownloadedVideoSegments = function(minSegments) {
    try {
        const dashMetrics = player.getDashMetrics();
        const requests = dashMetrics.getHttpRequests('video') || [];
        let count = 0;
        for (let i = 0; i < requests.length; i++) {
            if (requests[i].type === 'MediaSegment') {
                count++;
            }
        }
        return count >= minSegments;
    } catch (e) {
        return false;
    }
}



//Not working well
function getActualRepresentation() {
    var dashMetrics = player.getDashMetrics();
    var switchInfo = dashMetrics.getCurrentRepresentationSwitch('video');
    // return dashMetrics.getCurrentRepresentationSwitch('video').to;
    return switchInfo && switchInfo.to ? switchInfo.to : null;
}

//Not working well
function captureRepresentationSwitch() {
    if (typeof window.qualityChanges === 'undefined') {
        window.qualityChanges = [];
        player.on(dashjs.MediaPlayer.events.REPRESENTATION_SWITCH, function(e) {
            window.qualityChanges.push({
                time: player.time(),
                quality: e.to.index
            });
        });
    }
    return window.qualityChanges;
}


