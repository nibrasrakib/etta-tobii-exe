window.dataLayer = window.dataLayer || [];
// function gtag() {
// 	dataLayer.push(arguments);
// }
// gtag("js", new Date());
// gtag("config", "UA-153659007-2");

// document.addEventListener("DOMContentLoaded", function () {
//     const element = document.getElementById("math-tex");
//     // use back tick here to allow quotation marks query to not interfere with the syntax
//     const TeX = `\\text{Current query path: {{query|safe}}}`;
//     if (element) {
//         katex.render(TeX, element, {
//             throwOnError: false,
//         });
//     }
// });

const queryString = window.location.search;
const urlParams = new URLSearchParams(queryString);
const databaseOpt = urlParams.get("dataset_opt");

const currentUrl = window.location.href;

var encodedSignOutUrl = encodeURIComponent(currentUrl);
// if url has dataset_opt parameter
if (databaseOpt) {
	// if is searching user upload file
	if (databaseOpt.includes(".txt") || databaseOpt.includes(".csv")) {
		//console.log("searching user upload file")
		// encodedSignOutUrl = encodeURIComponent(homeUrl);
	}
}

if (login) {
	// document.getElementById("signin").innerHTML = `<i class="fa fa-sign-in"></i>  Sign out`;
	// // document.getElementById("signup").style.display="none"
	// if (loginType == "onyen") {
	// 	document.getElementById("signin").href =
	// 		"https://pattie.unc.edu/Shibboleth.sso/Logout?return=" +
	// 		encodedSignOutUrl;
	// 	document.getElementById("community_link").style.display = "block";
	// }
	// if (loginType == "general") {
	// 	document.getElementById("signin").href =
	// 		logoutUrl + "?target=" + encodedSignOutUrl;
	// }


	//if (stage === 'alpha') {
	//    document.getElementByClass("library-dropdown")[0].style.visibility = "visible";
	//}
} else {
	// document.getElementById("signin").innerHTML = '<i class="fa fa-sign-in"></i>  Sign in'

	/* redirect to login page on clicking sign in button*/
	// var encodedSignInSignUpUrl = encodeURIComponent(currentUrl)
	// document.getElementById("signin").href = {{url_for('login')|tojson}} + "?target=" + encodedSignInSignUpUrl;

	/* assign onyen sign in link to the sign in button*/
	/* get the target url to return to */
	// var encodedSignInSignUpUrl = encodeURIComponent(currentUrl);
	// const encodedTarget = encodedSignInSignUpUrl;
	// // const loginHandlerUrl = {{url_for('onyen_login_handler')|tojson}}
	// /* double level of encoding is needed for browser to correctly do redirection */
	// const encodedOnyenSignInUrl = encodeURIComponent(
	// 	loginHandlerUrl + "?target=" + encodedTarget
	// );



	// document.getElementById("signin").href =
	// 	"https://pattie.unc.edu/Shibboleth.sso/Login?target=" +
	// 	encodedOnyenSignInUrl;
	//console.log(encodedOnyenSignInUrl)

	// document.getElementById("signup").href = {{url_for('register')|tojson}} + "?target=" + encodedSignInSignUpUrl;
	//if (stage === 'alpha') {
	//    document.getElementsByClassName("library-dropdown")[0].style.display = "none";
	//}
}
