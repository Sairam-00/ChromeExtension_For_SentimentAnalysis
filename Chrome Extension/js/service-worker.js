var ws = null;
chrome.runtime.onConnect.addListener(function(port) {
    if(ws == null){
        ws = new WebSocket("ws://localhost:8080");
    }
    if(ws.readyState != 1){
        try{
            ws = new WebSocket("ws://localhost:8080");
        }catch(e){
            port.postMessage({
                from: "error",
            });
        }
    }
    port.onMessage.addListener(function(msg) {
        if(msg.from === "content"){
            var text = msg.text
            ws.send(text);
        }
        
    });
    ws.onmessage = function(event){
        port.postMessage({
            from: "service-worker",
            output: event.data
        });
    }
});

