$(function() {
    $('#sidenav a').bind('click',function(event){
        var $anchor = $(this);
 
        $('html, body').stop().animate({
            scrollTop: $($anchor.attr('href')).offset().top - 90
        }, 500);
        event.preventDefault();
    });
});