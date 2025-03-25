/*commented*/
/*Creates a stock input page*/
function create_stock_input(){
    /*Gets the div*/
    const adding_div = document.getElementById("div_of_stocks");
    var create_div = document.createElement("div");

    create_div.classList.add("stocks");
    //Creates the inputs
    const stock_code_input = document.createElement("input");
    const stock_amount_input = document.createElement("input");

    //Create the breaks
    var break1 = document.createElement("br");
    var break2 = document.createElement("br");

    //Sets the attributes for the inputs
    stock_code_input.classList.add("stock_codes");
    stock_code_input.id = "stock_code";
    stock_code_input.name = "stock_codes";
    stock_code_input.maxLength = 5;
    stock_code_input.minLength = 1;
    stock_amount_input.classList.add("stock_amounts");
    stock_amount_input.name = "stock_amounts";
    stock_amount_input.id = "stock_amount";
    stock_amount_input.setAttribute('type', 'number');

    //Creates the Text
    const stock_txt = document.createTextNode("Stock Code: ");
    const number_txt = document.createTextNode("Shares: ");
    const deletetion = document.createElement("input");

    //Creates the button and adds the function 
    deletetion.type = "button";
    deletetion.value = "Delete Stock";
    deletetion.onclick = function(){
        create_div.style.display = "none";
        create_div.remove();
    };

    //Styles the div
    create_div.style.backgroundColor = "#FFFFFF";
    create_div.style.border = "1px solid black";
    create_div.style.fontSize = "12px";

    //Adds it to the div
    create_div.appendChild(stock_txt);
    create_div.appendChild(stock_code_input);
    create_div.appendChild(break1);
    create_div.appendChild(number_txt);
    create_div.appendChild(stock_amount_input);
    create_div.appendChild(break2);
    create_div.appendChild(deletetion);

    //Adds final div to the page
    adding_div.appendChild(create_div);
}
/*
function color_stocks(){
    const stock_arr = document.getElementsByClassName("stocks");
    for (let i = 0; i < stock_arr.length; i++){
        if (stock_arr[i].getElementByID("stock_code") in failed_stocks){
            stock_arr[i].style.backgroundColor = "#f44336";
        }else{
            stock_arr[i].style.backgroundColor = "#4caf50";
        }
    }
}

function create_potential(stock){
    const potential_stock = document.getElementById("potential_stocks");
    var new_div = document.createElement("div");
    new_div.classList.add("potential_stocks"); 
    var stock_code = document.createTextNode(stock[0]);
    var company_name = document.createTextNode(stock[1]);
    var price = document.createTextNode(stock[3]);
    
    new_div.appendChild(stock_code);
    new_div.appendChild(company_name);
    new_div.appendChild(price);
    potential_stock.appendChild(new_div);
}
    */


