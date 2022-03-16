var pos = 60
! function() {
    var root = '<div id="inject_root" class="m-4"></div>'
    $(document.body).append(root);

    var port = chrome.runtime.connect({name: "ml"});
    var timeout = null;
    var old_text = "";
    $('[title="Search"]').keyup(function() {
        clearTimeout(timeout);
        var text = $(this).val();
        timeout = setTimeout(function() {
            if(old_text != text){
                old_text = text;
                try{
                port.postMessage({
                    from: "content",
                    text: text
                });
            }catch(e){}
            }
        }, 1000);
    })
    port.onMessage.addListener(function(msg) {
        if(msg.from ==="error"){
            alert("Python script not running")
        }else{
            var output = msg.output;
            injectHtml(output);
        }
    })

    function injectHtml(output){
        var time = new Date().toLocaleTimeString()
        var code = 
        `<div class="toast-container" style="position: absolute; top: `+pos+`px; right: 10px;">
        <div class="toast fade show">
            <div class="toast-header">
                <strong class="me-auto"><i class="bi-globe"></i>Text Sentiment</strong>
                <small>`+time+`</small>
                <button type="button" onclick="" class="btn-close" data-bs-dismiss="toast"></button>
            </div>
            <div class="toast-body">
                `+output+`
            </div>
        </div>`
        $('#inject_root').append(code).ready(function(){
            $(".toast").toast({
                autohide: false
            });
        });

    }



}()
