function updateDateTime() {
    const now = new Date();
    document.getElementById("dateTime").innerText = now.toLocaleString();
}
setInterval(updateDateTime, 1000);
updateDateTime();

document.getElementById("marathonForm").addEventListener("submit", (e) => {
    e.preventDefault();

    const name = name.value.trim();
    const gender = gender.value;
    const phone = phoneNumber.value.trim();
    const email = email.value.trim();
    const address = address.value.trim();
    const size = size.value;

    if (!name || !gender || !phone || !email || !address || !size) {
        alert("Please fill in all required fields!");
        return;
    }

    if (!/^\d{10}$/.test(phone)) {
        alert("Phone number must be exactly 10 digits.");
        return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
        alert("Please enter a valid email address.");
        return;
    }

    alert("Registration Successful! Thank you for joining Samarpana Marathon.");
    reset();
});
