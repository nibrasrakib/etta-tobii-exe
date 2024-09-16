function changeClusterNumber (value) {
	document.getElementById("num_cls").value = value;
}
function hideOption () {
	document.getElementById("cluster_option").style.display = "none";
}
function hideSelectClusterNum () {
	var cluster_num_select = document.getElementsByClassName("select")[1];
	cluster_num_select.style.display = "none";
}
function showSelectClusterNum () {
	var cluster_num_select = document.getElementsByClassName("select")[1];
	cluster_num_select.style.display = "inline-block";
}
document.getElementById("main_heading").style = "margin-top:10%";
//console.log(document.getElementById("num_cls").value)

// $('#inputfile').trigger('click')
//build dataset selecting menu
// var uploaded_files = {{uploaded_files|safe}}
//console.log(uploaded_files)
$(document).ready(function () {
	// var login = {{login|safe}}
	//console.log(document.URL);

	if (!login) {
		document.getElementById("delete").style.display = "none";
		document.getElementById("upload").style.display = "none";
		document.getElementById("dataset_opt").style.display = "none";
		document.getElementById("select_arrow").style.display = "none";
		document.getElementById("Explore").style.marginTop = "5vh";
		hideSelectClusterNum();
	}

	/* clear all session storage data when landing on home page, to search from fresh beginning */
	sessionStorage.clear();

	// configure redirect to cluster url (with args) on clicking search button
	$("#usrform").on("submit", function (e) {
		e.preventDefault();
		let pubmed_default_entity = "keywords";
		let expert_default_entity = "experts";
		let dataset = $("#dataset_opt").val();
		if (dataset == "NewsAPI") {
			window.location.assign(
				clusterUrl +
				"?dataset_opt=" +
				$("#dataset_opt").val() +
				"&query=" +
				$("#query-primary").val() +
				"&num_cls=" +
				$("#num_cls").val()
			);
		} else if (dataset == "Experts") {
			window.location.assign(
				clusterUrl +
				"?dataset_opt=" +
				$("#dataset_opt").val() +
				"&query=" +
				$("#query-primary").val() +
				"&entity=" +
				expert_default_entity
			);
		} else {

			window.location.assign(
				clusterUrl +
				"?dataset_opt=" +
				$("#dataset_opt").val() +
				"&query=" +
				$("#query-primary").val() +
				"&entity=" +
				pubmed_default_entity +
				"&num_cls=" +
				$("#num_cls").val()
			);
		}
	});

	// configure redirect to delete endpoint to execute the delete operation
	$("#delete").on("click", function (e) {
		e.preventDefault();
		window.location.replace(
			deleteFileUrl + "?file=" + $("#dataset_opt").val()
		);
	});
	//document.getElementById("tutorial").innerHTML= "<a href='{{url_for('static', filename='PATTIE_instrucitons.html')}}' class='text_link' onclick=\"return !window.open(this.href, 'tutorial', 'width=800,height=600')\" target=\"_blank\" style='text-align:left;margin-left:-1200px'>About</a>";

	var opt_check = 0;
	for (var i = 0; i < uploaded_files.length; i++) {
		createOption_dataset(uploaded_files[i]);
		//console.log("file i = " + uploaded_files[i])
		if (uploaded_files[i] == "Voicearch_utf8") {
			opt_check = 1;
		}
	}
	var cur_value = document.getElementById("dataset_opt").value;
	//console.log(cur_value)
	changeField(cur_value);

	//change the default value of options according to the previous page
	var prev_page = document.referrer;
	if (prev_page.indexOf("voicearch") >= 0) {
		//console.log("that's what we want")
		if (opt_check == 1) {
			document.getElementById("dataset_opt").value = "Voicearch_utf8";
		}
		//console.log(opt_check)
	}
});

function redirectUpload () {
	window.location.href = document.URL + "upload";
}

function changeField (value) {
	if (value != "Experts") {
		if (login) {
			showSelectClusterNum();
		}
	}

	// default option: if not select uploaded file, hide delete button
	document.getElementById("delete").style.visibility = "hidden";
	if (value == "PLOS") {
		document.getElementById("date_div").innerHTML = "";

		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
		createOption_field("Title");
		createOption_field("Abstract");
		createOption_field("Body");
	} else if (value == "Upload...") {
		document.getElementById("date_div").innerHTML = "";

		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
		createOption_field("Title");
		createOption_field("Body");
		$("#upload").click();
	} else if (value == "NYTIMES") {
		document.getElementById("date_div").innerHTML = "";

		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
		createOption_field("Title");
		createOption_field("Body");
	} else if (value == "DIABETES") {
		document.getElementById("date_div").innerHTML = "";

		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
		createOption_field("Title");
		createOption_field("Abstract");
	} else if (value == "PubMedAPI") {
		var note_div = document.createElement("div");
		note_div.setAttribute("id", "note_div");
		note_div.setAttribute(
			"style",
			"text-align: left; padding-left:27.6%; font-size: 14px; font-style:italic;"
		);
		//document.getElementById('date_div').appendChild(note_div);
		//document.getElementById('note_div').innerHTML='Source: PubMed';
		createOption_field("All fields");
	} else if (value == "Experts") {
		hideSelectClusterNum();
		note_div.setAttribute("id", "note_div");
		note_div.setAttribute(
			"style",
			"text-align: left; padding-left:27.6%; font-size: 14px; font-style:italic;"
		);
		//document.getElementById('date_div').appendChild(note_div);
		//document.getElementById('note_div').innerHTML='Source: Academic Experts';
		//createOption_field('All fields');
	} else if (value == "GoogleAPI") {
		document.getElementById("date_div").innerHTML = "";

		var note_div = document.createElement("div");
		note_div.setAttribute("id", "note_div");
		note_div.setAttribute(
			"style",
			"text-align: left; padding-left:27.6%; font-size: 14px; font-style:italic;"
		);
		document.getElementById("date_div").appendChild(note_div);
		document.getElementById("note_div").innerHTML =
			"Sources: ACM Digital Library, ASIS&T Digital Library, IEEE Xplore";
		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
	} else if (value == "GoogleAPI2") {
		document.getElementById("date_div").innerHTML = "";

		var note_div = document.createElement("div");
		note_div.setAttribute("id", "note_div");
		note_div.setAttribute(
			"style",
			"text-align: left; padding-left:27.6%; font-size: 14px; font-style:italic;"
		);
		document.getElementById("date_div").appendChild(note_div);
		document.getElementById("note_div").innerHTML =
			"Sources: all domains with .gov or .edu";
		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
	} else if (value == "GoogleAPI3") {
		document.getElementById("date_div").innerHTML = "";

		var note_div = document.createElement("div");
		note_div.setAttribute("id", "note_div");
		note_div.setAttribute(
			"style",
			"text-align: left; padding-left:27.6%; font-size: 14px; font-style:italic;"
		);
		document.getElementById("date_div").appendChild(note_div);
		document.getElementById("note_div").innerHTML =
			"Sources: Nature, Science, Cell, PNAS";
		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
	} else if (value == "GoogleAPI4") {
		document.getElementById("date_div").innerHTML = "";

		var note_div = document.createElement("div");
		note_div.setAttribute("id", "note_div");
		note_div.setAttribute(
			"style",
			"text-align: left; padding-left:27.6%; font-size: 14px; font-style:italic;"
		);
		document.getElementById("date_div").appendChild(note_div);
		document.getElementById("note_div").innerHTML =
			"Source: PubMed Central";
		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
	} else if (value == "GoogleAPI5") {
		document.getElementById("date_div").innerHTML = "";

		var note_div = document.createElement("div");
		note_div.setAttribute("id", "note_div");
		note_div.setAttribute(
			"style",
			"text-align: left; padding-left:27.6%; font-size: 14px; font-style:italic;"
		);
		document.getElementById("date_div").appendChild(note_div);
		document.getElementById("note_div").innerHTML =
			"Source: CDC, WHO, Medline Plus";
		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
	} else if (value == "NewsAPI") {
		document.getElementById("date_div").innerHTML = "";

		document.getElementById("date_div").innerHTML += "Start Date: ";

		var x = document.createElement("INPUT");
		x.setAttribute("type", "date");
		x.setAttribute("name", "start_date");
		x.setAttribute("id", "start_date");
		x.setAttribute("value", "{{start_date}}");
		x.setAttribute("style", "opacity:0.75;background:DodgerBlue");
		document.getElementById("date_div").appendChild(x);

		var dt = document.createElement("dt");
		//document.getElementById('date_div').appendChild(dt);

		document.getElementById("date_div").innerHTML += "\nEnd Date: ";

		var x = document.createElement("INPUT");
		x.setAttribute("type", "date");
		x.setAttribute("name", "end_date");
		x.setAttribute("id", "end_date");
		x.setAttribute("value", "{{end_date}}");
		x.setAttribute("style", "opacity:0.9;background:DodgerBlue");
		document.getElementById("date_div").appendChild(x);

		document.getElementById("date_div").appendChild(dt);

		var note_div = document.createElement("div");
		note_div.setAttribute("id", "note_div");
		note_div.setAttribute(
			"style",
			"text-align: left; padding-left:27.6%; font-size: 14px; font-style:italic;"
		);
		document.getElementById("date_div").appendChild(note_div);
		document.getElementById("note_div").innerHTML =
			"Note:<dt>Please set your start date <dt>no earlier than 30 days from today. <dt>If you do not enter a search, the <dt>top 100 trending news from the last <dt>15 minutes will be retrieved.";

		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
	} else {
		//document.getElementById('date_div').innerHTML = ''
		//console.log(value)
		document.getElementById("field").innerHTML = "";
		createOption_field("All fields");
		// user has selected an uploaded file, reveal the delete button and enable delete file function
		document.getElementById("delete").style.visibility = "visible";
	}
}

function createOption_field (value) {
	el = document.createElement("option");
	el.value = value;
	el.innerHTML = value;
	el.id = value;
	document.getElementById("field").appendChild(el);
}

function createOption_dataset (value) {
	el = document.createElement("option");
	el.value = value;
	el.innerHTML = value;
	el.id = value;
	document.getElementById("dataset_opt").appendChild(el);
}

function deleteOption_dataset (value) {
	el = document.createElement("option");
	el.value = value;
	el.innerHTML = value;
	el.id = value;
	document.getElementById("dataset_opt").appendChild(el);
}

function changeClusters (value) {
	'<%Session["num_cls"] = "' + value + '"; %>';
}
